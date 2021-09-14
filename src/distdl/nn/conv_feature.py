import numpy as np
import torch
import torch.nn.functional as F

from distdl.nn.broadcast import Broadcast
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.mixins.conv_mixin import ConvMixin
from distdl.nn.mixins.halo_mixin import HaloMixin
from distdl.nn.module import Module
from distdl.utilities.slicing import assemble_slices
from distdl.utilities.torch import distdl_padding_to_torch_padding
from distdl.utilities.torch import TensorStructure
from distdl.utilities.torch import zero_volume_tensor


class DistributedFeatureConvBase(Module, HaloMixin, ConvMixin):
    r"""A feature-space partitioned distributed convolutional layer.

    This class provides the user interface to a distributed convolutional
    layer, where the input (and output) tensors are partitioned in
    feature-space only.

    The base unit of work is given by the input/output tensor partition.  This
    class requires the following of the tensor partitions:

    1. :math:`P_x` over input tensor :math:`x` has shape :math:`1 \times
       1 \times P_{d-1} \times \dots \times P_0`.

    The output partition, :math:`P_y`, is assumed to be the same as the
    input partition.

    The first dimension of the input/output partitions is the batch
    dimension,the second is the channel dimension, and remaining dimensions
    are feature dimensions.

    The learnable weight and bias terms does not have their own partition.
    They is stored at the 0th rank (index :math:`(0, 0,\dots, 0)` of
    :math:`P_x`.  Each worker in :math:`P_x` does have their own local
    convolutional layer but only one worker has learnable coefficients.

    All inputs to this base class are passed through to the underlying PyTorch
    convolutional layer.

    Parameters
    ----------
    P_x :
        Partition of input tensor.
    in_channels :
        (int)
        Number of channels in the input image
    out_channels :
        (int)
        Number of channels produced by the convolution
    kernel_size :
        (int or tuple)
        Size of the convolving kernel
    stride :
        (int or tuple, optional)
        Stride of the convolution. Default: 1
    padding :
        (int or tuple, optional)
        Zero-padding added to both sides of the input. Default: 0
    padding_mode :
        (string, optional)
        'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    dilation :
        (int or tuple, optional)
        Spacing between kernel elements. Default: 1
    groups :
        (int, optional)
        Number of blocked connections from input channels to output channels. Default: 1
    bias :
        (bool, optional)
        If True, adds a learnable bias to the output. Default: True
    buffer_manager :
        (BufferManager, optional)
        DistDL BufferManager. Default: None
    """

    # Convolution class for base unit of work.
    TorchConvType = None
    # Number of dimensions of a feature
    num_dimensions = None

    def __init__(self,
                 P_x,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 padding_mode='zeros',
                 dilation=1,
                 groups=1,
                 bias=True,
                 buffer_manager=None):

        super(DistributedFeatureConvBase, self).__init__()

        # P_x is 1 x 1 x P_d-1 x ... x P_0
        self.P_x = P_x

        # Back-end specific buffer manager for economic buffer allocation
        if buffer_manager is None:
            buffer_manager = self._distdl_backend.BufferManager()
        elif type(buffer_manager) is not self._distdl_backend.BufferManager:
            raise ValueError("Buffer manager type does not match backend.")
        self.buffer_manager = buffer_manager

        if not self.P_x.active:
            return

        dims = len(self.P_x.shape)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._expand_parameter(kernel_size)
        self.stride = self._expand_parameter(stride)
        self.padding = self._expand_parameter(padding)
        self.padding_mode = padding_mode
        self.dilation = self._expand_parameter(dilation)
        self.groups = groups
        self.use_bias = bias

        self.serial = self.P_x.size == 1

        if self.serial:
            self.conv_layer = self.TorchConvType(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 padding_mode=self.padding_mode,
                                                 dilation=self.dilation,
                                                 groups=self.groups,
                                                 bias=self.use_bias)
            self.weight = self.conv_layer.weight
            self.bias = self.conv_layer.bias
        else:
            self.conv_layer = self.TorchConvType(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=0,
                                                 padding_mode='zeros',
                                                 dilation=self.dilation,
                                                 groups=groups,
                                                 bias=bias)

        if self.serial:
            return

        # We will be using global padding to compute local padding,
        # so expand it to a numpy array
        global_padding = np.pad(self.padding,
                                pad_width=(dims-len(self.padding), 0),
                                mode='constant',
                                constant_values=0)
        self.global_padding = global_padding

        pad_left_right = self.global_padding.reshape((dims, 1)) + np.zeros((dims, 2), dtype=np.int)
        self.local_padding = self._compute_local_padding(pad_left_right)

        # Weights and biases partition
        P_wb = self.P_x.create_partition_inclusive([0])
        self.P_wb_cart = P_wb.create_cartesian_topology_partition([1])

        # Release temporary resources
        P_wb.deactivate()

        # We want only the root rank of the broadcast to have a weight and a
        # bias parameter. Every other rank gets a zero-volume tensor.
        if self.P_wb_cart.active:
            self.weight = torch.nn.Parameter(self.conv_layer.weight.detach())

            if self.conv_layer.bias is not None:
                self.bias = torch.nn.Parameter(self.conv_layer.bias.detach())
        else:
            self.register_buffer('weight', zero_volume_tensor())

            if self.conv_layer.bias is not None:
                self.register_buffer('bias', zero_volume_tensor())

        self.weight.requires_grad = self.conv_layer.weight.requires_grad

        if self.conv_layer.bias is not None:
            self.bias.requires_grad = self.conv_layer.bias.requires_grad

        # https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677/2
        new_weight = self.conv_layer.weight.detach() * 0
        new_weight.requires_grad = self.conv_layer.weight.requires_grad
        del self.conv_layer.weight
        self.conv_layer.weight = new_weight

        if self.conv_layer.bias is not None:
            new_bias = self.conv_layer.bias.detach() * 0
            new_bias.requires_grad = self.conv_layer.bias.requires_grad
            del self.conv_layer.bias
            self.conv_layer.bias = new_bias

        self.w_broadcast = Broadcast(self.P_wb_cart, self.P_x,
                                     preserve_batch=False)

        if self.conv_layer.bias is not None:
            self.b_broadcast = Broadcast(self.P_wb_cart, self.P_x,
                                         preserve_batch=False)

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
        r"""Distributed (feature) convolution module setup function.

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

        if self.serial:
            return

        # Compute global and local shapes with padding
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
        r"""Distributed (channel) convolution module teardown function.

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

        if self.serial:
            return self.conv_layer(input)

        w = self.w_broadcast(self.weight)
        self.conv_layer.weight = w

        if self.conv_layer.bias is not None:
            b = self.b_broadcast(self.bias)
            self.conv_layer.bias = b

        # Compute the total padding and convert to PyTorch format
        total_padding = self.local_padding + self.halo_shape
        torch_padding = distdl_padding_to_torch_padding(total_padding)

        if total_padding.sum() == 0:
            input_padded = input
        else:
            pad_mode = 'constant' if self.padding_mode == 'zeros' else self.padding_mode
            input_padded = F.pad(input, pad=torch_padding, mode=pad_mode, value=0)

        input_exchanged = self.halo_layer(input_padded)
        input_needed = input_exchanged[self.needed_slices]
        conv_output = self.conv_layer(input_needed)
        return conv_output


class DistributedFeatureConv1d(DistributedFeatureConvBase):
    r"""A feature-partitioned distributed 1d convolutional layer.

    """

    TorchConvType = torch.nn.Conv1d
    num_dimensions = 1


class DistributedFeatureConv2d(DistributedFeatureConvBase):
    r"""A feature-partitioned distributed 2d convolutional layer.

    """

    TorchConvType = torch.nn.Conv2d
    num_dimensions = 2


class DistributedFeatureConv3d(DistributedFeatureConvBase):
    r"""A feature-partitioned distributed 3d convolutional layer.

    """

    TorchConvType = torch.nn.Conv3d
    num_dimensions = 3
