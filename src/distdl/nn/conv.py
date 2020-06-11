import numpy as np
import torch

from distdl.nn.broadcast import Broadcast
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.halo_mixin import HaloMixin
from distdl.nn.module import Module
from distdl.nn.padnd import PadNd
from distdl.nn.unpadnd import UnPadNd
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

    def __init__(self, x_in_sizes, P_cart, *args, **kwargs):

        super(DistributedConvBase, self).__init__()

        self.x_in_sizes = x_in_sizes
        self.P_cart = P_cart

        if not self.P_cart.active:
            return

        # Do this before checking serial so that the layer works properly
        # in the serial case
        self.conv_layer = self.TorchConvType(*args, **kwargs)

        self.serial = False
        if self.P_cart.size == 1:
            self.serial = True
            return

        # Weights and biases partition
        self.P_wb = self.P_cart.create_partition_inclusive([0])
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

        self.w_broadcast = Broadcast(self.P_wb_cart, self.P_cart)

        if self.conv_layer.bias is not None:
            self.b_broadcast = Broadcast(self.P_wb_cart, self.P_cart)

        self.halo_sizes, self.recv_buffer_sizes, self.send_buffer_sizes, self.needed_ranges = \
            self._compute_exchange_info(self.x_in_sizes,
                                        self.conv_layer.kernel_size,
                                        self.conv_layer.stride,
                                        self.conv_layer.padding,
                                        self.conv_layer.dilation,
                                        self.P_cart.active,
                                        self.P_cart.dims,
                                        self.P_cart.coords)

        self.halo_sizes = self.halo_sizes.astype(int)
        self.needed_ranges = self.needed_ranges.astype(int)

        # Unpad sizes are padding in the dimensions where we have a halo,
        # otherwise 0
        self.pads = np.concatenate(([0, 0], self.conv_layer.padding))
        self.unpad_sizes = []
        for pad, halo_size in zip(self.pads, self.halo_sizes):
            self.unpad_sizes.append(np.where(halo_size > 0, pad, 0))
        self.unpad_sizes = np.asarray(self.unpad_sizes)

        self.needed_slices = assemble_slices(self.needed_ranges[:, 0], self.needed_ranges[:, 1])

        self.pad_layer = PadNd(self.halo_sizes, value=0, partition=self.P_cart)
        self.unpad_layer = UnPadNd(self.unpad_sizes, value=0, partition=self.P_cart)

        self.local_x_in_sizes_padded = self._compute_local_x_in_sizes_padded(self.x_in_sizes,
                                                                             self.P_cart.dims,
                                                                             self.P_cart.coords,
                                                                             self.halo_sizes)

        self.halo_layer = HaloExchange(self.local_x_in_sizes_padded,
                                       self.halo_sizes,
                                       self.recv_buffer_sizes,
                                       self.send_buffer_sizes,
                                       self.P_cart)

    def forward(self, input):

        if not self.P_cart.active:
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
