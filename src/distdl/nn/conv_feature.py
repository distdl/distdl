import numpy as np
import torch

from distdl.nn.broadcast import Broadcast
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.mixins.conv_mixin import ConvMixin
from distdl.nn.mixins.halo_mixin import HaloMixin
from distdl.nn.module import Module
from distdl.nn.padnd import PadNd
from distdl.nn.unpadnd import UnpadNd
from distdl.utilities.slicing import assemble_slices
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
       P_{\text{c_in}} \times 1 \times \dots \times 1`.

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

    """

    # Convolution class for base unit of work.
    TorchConvType = None

    def __init__(self, P_x, *args, **kwargs):

        super(DistributedFeatureConvBase, self).__init__()

        # P_x is 1 x 1 x P_d-1 x ... x P_0
        self.P_x = P_x

        if not self.P_x.active:
            return

        # Do this before checking serial so that the layer works properly
        # in the serial case
        self.conv_layer = self.TorchConvType(*args, **kwargs)

        self.serial = False
        if self.P_x.size == 1:
            self.serial = True
            return

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
            self.weight = zero_volume_tensor()

            if self.conv_layer.bias is not None:
                self.bias = zero_volume_tensor()

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

        # We need the halo shape, and other info, to fully populate the pad,
        # halo exchange, and unpad layers.  For pad and unpad, we defer their
        # construction to the pre-forward hook.
        self.pad_layer = None
        self.unpad_layer = None

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

        # To compute the halo regions, we need the global tensor shape.  This
        # is not available until when the input is provided.
        x_global_structure = \
            self._distdl_backend.assemble_global_tensor_structure(input[0], self.P_x)

        # Using that information, we can get there rest of the halo information
        exchange_info = self._compute_exchange_info(x_global_structure.shape,
                                                    self.conv_layer.kernel_size,
                                                    self.conv_layer.stride,
                                                    self.conv_layer.padding,
                                                    self.conv_layer.dilation,
                                                    self.P_x.active,
                                                    self.P_x.shape,
                                                    self.P_x.index)
        halo_shape = exchange_info[0]
        recv_buffer_shape = exchange_info[1]
        send_buffer_shape = exchange_info[2]
        needed_ranges = exchange_info[3]

        # Now we have enough information to instantiate the padding shim
        self.pad_layer = PadNd(halo_shape, value=0)

        # We can also set up part of the halo layer.
        self.halo_layer = HaloExchange(self.P_x,
                                       halo_shape,
                                       recv_buffer_shape,
                                       send_buffer_shape)

        # We have to select out the "unused" entries.  Sometimes there can
        # be "negative" halos.
        self.needed_slices = assemble_slices(needed_ranges[:, 0],
                                             needed_ranges[:, 1])

        # Unpad shape are conv layer's padding in the dimensions where we have
        # a halo, otherwise 0.  There is no halo in the batch and channel
        # dimensions.
        conv_padding = np.concatenate(([0, 0], self.conv_layer.padding))
        unpad_shape = []
        for pad, halo in zip(conv_padding, halo_shape):
            unpad_shape.append(np.where(halo > 0, pad, 0))
        unpad_shape = np.asarray(unpad_shape)

        self.unpad_layer = UnpadNd(unpad_shape, value=0)

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
        self.pad_layer = None
        self.unpad_layer = None
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

        input_padded = self.pad_layer(input)
        input_exchanged = self.halo_layer(input_padded)
        input_needed = input_exchanged[self.needed_slices]
        conv_output = self.conv_layer(input_needed)
        return self.unpad_layer(conv_output)


class DistributedFeatureConv1d(DistributedFeatureConvBase):
    r"""A feature-partitioned distributed 1d convolutional layer.

    """

    TorchConvType = torch.nn.Conv1d


class DistributedFeatureConv2d(DistributedFeatureConvBase):
    r"""A feature-partitioned distributed 2d convolutional layer.

    """

    TorchConvType = torch.nn.Conv2d


class DistributedFeatureConv3d(DistributedFeatureConvBase):
    r"""A feature-partitioned distributed 3d convolutional layer.

    """

    TorchConvType = torch.nn.Conv3d
