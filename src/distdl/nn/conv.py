import numpy as np
import torch

from distdl.nn.broadcast import Broadcast
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.halo_mixin import HaloMixin
from distdl.nn.module import Module
from distdl.nn.padnd import PadNd
from distdl.nn.unpadnd import UnpadNd
from distdl.utilities.slicing import assemble_slices
from distdl.utilities.torch import NoneTensor


class ConvMixin:

    def _compute_min_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take stride and dilation into account
        # padding might also not be correct in these cases...
        kernel_offsets = (kernel_sizes - 1) / 2

        # for even sized kernels, always shortchange the left side
        kernel_offsets[kernel_sizes % 2 == 0] -= 1

        bases = idx + kernel_offsets - pads
        return bases - kernel_offsets

    def _compute_max_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take stride and dilation into account
        # padding might also not be correct in these cases...
        kernel_offsets = (kernel_sizes - 1) / 2

        bases = idx + kernel_offsets - pads
        return bases + kernel_offsets


class DistributedConvBase(Module, HaloMixin, ConvMixin):

    TorchConvType = None

    def __init__(self, P_x, *args, **kwargs):

        super(DistributedConvBase, self).__init__()

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
        self.P_wb = self.P_x.create_partition_inclusive([0])
        self.P_wb_cart = self.P_wb.create_cartesian_topology_partition([1])

        # We want only the root rank of the broadcast to have a weight and a bias parameter.
        # Every other rank gets a NoneTensor.
        if self.P_wb_cart.active:
            self.weight = torch.nn.Parameter(self.conv_layer.weight.detach())

            if self.conv_layer.bias is not None:
                self.bias = torch.nn.Parameter(self.conv_layer.bias.detach())

        else:
            self.weight = NoneTensor()

            if self.conv_layer.bias is not None:
                self.bias = NoneTensor()

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

        self.w_broadcast = Broadcast(self.P_wb_cart, self.P_x)

        if self.conv_layer.bias is not None:
            self.b_broadcast = Broadcast(self.P_wb_cart, self.P_x)

        # We need the halo sizes, and other info, to fully populate the pad,
        # halo exchange, and unpad layers.  For pad and unpad, we defer their
        # construction to the pre-forward hook.

        self.pad_layer = None
        self.unpad_layer = None

        # We need to be able to remove some data from the input to the conv
        # layer.
        self.needed_slices = None

        # For the halo layer we also defer construction, so that we can have
        # the halo sizes for the input.  The halo will allocate its own
        # buffers, but it needs this information at construction to be able
        # to do this in the pre-forward hook.

        self.halo_layer = None

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_shape = None
        self._input_requires_grad = None

    def _distdl_module_setup(self, input):

        if not self.P_x.active:
            return

        if self.serial:
            return

        global_tensor_sizes = self._distdl_backend.compute_global_tensor_sizes(input[0],
                                                                               self.P_x)
        exchange_info = self._compute_exchange_info(global_tensor_sizes,
                                                    self.conv_layer.kernel_size,
                                                    self.conv_layer.stride,
                                                    self.conv_layer.padding,
                                                    self.conv_layer.dilation,
                                                    self.P_x.active,
                                                    self.P_x.dims,
                                                    self.P_x.coords)
        halo_sizes = exchange_info[0]
        recv_buffer_sizes = exchange_info[1]
        send_buffer_sizes = exchange_info[2]
        needed_ranges = exchange_info[3]

        # Now we have enough information to instantiate the padding shim
        self.pad_layer = PadNd(halo_sizes, value=0)

        # We can also set up part of the halo layer.
        self.halo_layer = HaloExchange(self.P_x,
                                       halo_sizes,
                                       recv_buffer_sizes,
                                       send_buffer_sizes)

        # We have to select out the "unused" entries.
        self.needed_slices = assemble_slices(needed_ranges[:, 0],
                                             needed_ranges[:, 1])

        # Unpad sizes are conv layer's padding in the dimensions where we have
        # a halo, otherwise 0.  There is no halo in the batch and channel
        # dimensions.
        conv_padding = np.concatenate(([0, 0], self.conv_layer.padding))
        unpad_sizes = []
        for pad, halo_size in zip(conv_padding, halo_sizes):
            unpad_sizes.append(np.where(halo_size > 0, pad, 0))
        unpad_sizes = np.asarray(unpad_sizes)

        self.unpad_layer = UnpadNd(unpad_sizes, value=0)

        self._distdl_is_setup = True
        self._input_shape = input[0].shape
        self._input_requires_grad = input[0].requires_grad

    def _distdl_module_teardown(self, input):

        # Reset all sub_layers
        self.pad_layer = None
        self.unpad_layer = None
        self.needed_slices = None
        self.halo_layer = None

        # Reset any info about the input
        self._distdl_is_setup = False
        self._input_shape = None
        self._input_requires_grad = None

    def _distdl_input_changed(self, input):

        if input[0].requires_grad != self._input_requires_grad:
            return True

        if input[0].shape != self._input_shape:
            return True

        return False

    def forward(self, input):

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


class DistributedConv1d(DistributedConvBase):

    TorchConvType = torch.nn.Conv1d


class DistributedConv2d(DistributedConvBase):

    TorchConvType = torch.nn.Conv2d


class DistributedConv3d(DistributedConvBase):

    TorchConvType = torch.nn.Conv3d
