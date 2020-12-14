import numpy as np
import torch
import torch.nn.functional as F

from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.mixins.halo_mixin import HaloMixin
from distdl.nn.mixins.pooling_mixin import PoolingMixin
from distdl.nn.module import Module
from distdl.utilities.slicing import assemble_slices
from distdl.utilities.torch import TensorStructure


class DistributedPoolBase(Module, HaloMixin, PoolingMixin):
    r"""A feature-space partitioned distributed pooling layer.

    This class provides the user interface to a distributed pooling
    layer, where the input (and output) tensors are partitioned in
    feature-space only.

    The base unit of work is given by the input/output tensor partition.  This
    class requires the following of the tensor partitions:

    1. :math:`P_x` over input tensor :math:`x` has shape :math:`1 \times
       P_{\text{c_in}} \times 1 \times \dots \times 1`.

    The output partition, :math:`P_y`, is assumed to be the same as the
    input partition.

    The first dimension of the input/output partitions is the batch
    dimension,the second is the channel dimension, and remaining dimensions
    are feature dimensions.

    There are no learnable parameters.

    All inputs to this base class are passed through to the underlying PyTorch
    pooling layer.

    Parameters
    ----------
    P_x :
        Partition of input tensor.
    kernel_size :
        (int or tuple)
        The size of the window to take a max over.
    stride :
        (int or tuple, optional)
        Stride of the convolution.  Default: kernel_size
    padding :
        (int or tuple, optional)
        Zero-padding added to both sides of the input.  Default: 0
    dilation :
        (int or tuple, optional)
        A parameter that controls the stride of elements in the window.  Default: 1

        .. warning::
        Dilation is only supported on MaxPooling layers.
    buffer_manager :
        (BufferManager, optional)
        DistDL BufferManager. Default: None

    """

    # Pooling class for base unit of work.
    TorchPoolType = None  # noqa F821
    # Number of dimensions of a feature
    num_dimensions = None

    def __init__(self,
                 P_x,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 buffer_manager=None):

        super(DistributedPoolBase, self).__init__()

        # P_x is 1 x 1 x P_d-1 x ... x P_0
        self.P_x = P_x

        self.is_avg = self.TorchPoolType in [torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d]

        # Back-end specific buffer manager for economic buffer allocation
        if buffer_manager is None:
            buffer_manager = self._distdl_backend.BufferManager()
        elif type(buffer_manager) is not self._distdl_backend.BufferManager:
            raise ValueError("Buffer manager type does not match backend.")
        self.buffer_manager = buffer_manager

        if not self.P_x.active:
            return

        dims = len(self.P_x.shape)

        self.kernel_size = self._expand_parameter(kernel_size)
        self.stride = self._expand_parameter(stride)
        self.padding = self._expand_parameter(padding)
        self.dilation = self._expand_parameter(dilation)

        if self.is_avg and not all(x == 1 for x in self.dilation):
            raise ValueError('dilation is only supported for MaxPooling layers.')

        # PyTorch does not support dilation for AvgPooling layers
        if self.is_avg:
            self.pool_layer = self.TorchPoolType(kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=0)
        else:
            self.pool_layer = self.TorchPoolType(kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=0,
                                                 dilation=self.dilation)

        # We will be using global padding to compute local padding,
        # so expand it to a numpy array
        global_padding = np.pad(self.padding,
                                pad_width=(dims-len(self.padding), 0),
                                mode='constant',
                                constant_values=0)
        self.global_padding = global_padding

        pad_left_right = self.global_padding.reshape((dims, 1)) + np.zeros((dims, 2), dtype=np.int)
        self.local_padding = self._compute_local_padding(pad_left_right)

        # We need to be able to remove some data from the input to the conv
        # layer.
        self.needed_slices = None

        # For the halo layer we also defer construction, so that we can have
        # the halo shape for the input.  The halo will allocate its own
        # buffers, but it needs this information at construction to be able
        # to do this in the pre-forward hook.
        self.halo_layer = None

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

    def _expand_parameter(self, param):
        # If the given input is not of size num_dimensions, expand it so.
        # If not possible, raise an exception.
        param = np.atleast_1d(param)
        if len(param) == 1:
            param = np.ones(self.num_dimensions, dtype=int) * param[0]
        elif len(param) == self.num_dimensions:
            pass
        else:
            raise ValueError('Invalid parameter: ' + str(param))
        return tuple(param)

    def _distdl_module_setup(self, input):
        r"""Distributed (feature) pooling module setup function.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        self._distdl_is_setup = True
        self._input_tensor_structure = TensorStructure(input[0])

        if not self.P_x.active:
            return

        # To compute the halo regions, we need the global tensor shape.  This
        # is not available until when the input is provided.
        x_global_structure = \
            self._distdl_backend.assemble_global_tensor_structure(input[0], self.P_x)
        x_local_structure = TensorStructure(input[0])
        x_global_shape = x_global_structure.shape
        x_local_shape = x_local_structure.shape
        x_global_shape_after_pad = x_global_shape + 2*self.global_padding
        x_local_shape_after_pad = x_local_shape + np.sum(self.local_padding, axis=1, keepdims=False)
        x_local_structure_after_pad = TensorStructure(input[0])
        x_local_structure_after_pad.shape = x_local_shape_after_pad

        # We need to compute the halos with respect to the explicit padding.
        # So, we assume the padding is already added, then compute the halo regions.
        compute_subtensor_shapes_unbalanced = \
            self._distdl_backend.tensor_decomposition.compute_subtensor_shapes_unbalanced
        subtensor_shapes = \
            compute_subtensor_shapes_unbalanced(x_local_structure_after_pad, self.P_x)

        # Using that information, we can get there rest of the halo information
        exchange_info = self._compute_exchange_info(x_global_shape_after_pad,
                                                    self.kernel_size,
                                                    self.stride,
                                                    self._expand_parameter(0),
                                                    self.dilation,
                                                    self.P_x.active,
                                                    self.P_x.shape,
                                                    self.P_x.index,
                                                    subtensor_shapes=subtensor_shapes)
        halo_shape = exchange_info[0]
        recv_buffer_shape = exchange_info[1]
        send_buffer_shape = exchange_info[2]
        needed_ranges = exchange_info[3]

        self.halo_shape = halo_shape

        # We can also set up part of the halo layer.
        self.halo_layer = HaloExchange(self.P_x,
                                       halo_shape,
                                       recv_buffer_shape,
                                       send_buffer_shape,
                                       buffer_manager=self.buffer_manager)

        # We have to select out the "unused" entries.  Sometimes there can
        # be "negative" halos.
        self.needed_slices = assemble_slices(needed_ranges[:, 0],
                                             needed_ranges[:, 1])

    def _distdl_module_teardown(self, input):
        r"""Distributed (channel) pooling module teardown function.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        # Reset all sub_layers
        self.needed_slices = None
        self.halo_layer = None

        # Reset any info about the input
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

    def _distdl_input_changed(self, input):
        r"""Determine if the structure of inputs has changed.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        new_tensor_structure = TensorStructure(input[0])

        return self._input_tensor_structure != new_tensor_structure

    def _to_torch_padding(self, pad):
        r"""
        Accepts a NumPy ndarray describing the padding, and produces the torch F.pad format:
            [[a_0, b_0], ..., [a_n, b_n]]  ->  (a_n, b_n, ..., a_0, b_0)

        """
        return tuple(np.array(list(reversed(pad)), dtype=int).flatten())

    def _compute_local_padding(self, padding):
        r"""
        Computes the amount of explicit padding required on the current rank,
        given the global padding.

        """
        should_pad_left = [k == 0 for k in self.P_x.index]
        should_pad_right = [k == d-1 for k, d in zip(self.P_x.index, self.P_x.shape)]
        should_pad = np.stack((should_pad_left, should_pad_right), axis=1)
        local_padding = np.where(should_pad, padding, 0)
        return local_padding

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to be broadcast.

        """

        if not self.P_x.active:
            return input.clone()

        # Compute the total padding and convert to PyTorch format
        total_padding = self.local_padding + self.halo_shape
        torch_padding = self._to_torch_padding(total_padding)

        if total_padding.sum() == 0:
            input_padded = input
        else:
            input_padded = F.pad(input, pad=torch_padding, mode='constant', value=0)

        input_exchanged = self.halo_layer(input_padded)
        input_needed = input_exchanged[self.needed_slices]
        pool_output = self.pool_layer(input_needed)
        return pool_output


class DistributedAvgPool1d(DistributedPoolBase):
    r"""A feature-partitioned distributed 1d average pooling layer.

    """

    TorchPoolType = torch.nn.AvgPool1d
    num_dimensions = 1


class DistributedAvgPool2d(DistributedPoolBase):
    r"""A feature-partitioned distributed 2d average pooling layer.

    """

    TorchPoolType = torch.nn.AvgPool2d
    num_dimensions = 2


class DistributedAvgPool3d(DistributedPoolBase):
    r"""A feature-partitioned distributed 3d average pooling layer.

    """

    TorchPoolType = torch.nn.AvgPool3d
    num_dimensions = 3


class DistributedMaxPool1d(DistributedPoolBase):
    r"""A feature-partitioned distributed 1d max pooling layer.

    """

    TorchPoolType = torch.nn.MaxPool1d
    num_dimensions = 1


class DistributedMaxPool2d(DistributedPoolBase):
    r"""A feature-partitioned distributed 2d max pooling layer.

    """

    TorchPoolType = torch.nn.MaxPool2d
    num_dimensions = 2


class DistributedMaxPool3d(DistributedPoolBase):
    r"""A feature-partitioned distributed 3d max pooling layer.

    """

    TorchPoolType = torch.nn.MaxPool3d
    num_dimensions = 3
