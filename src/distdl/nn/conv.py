import torch

from distdl.nn.broadcast import Broadcast
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.halo_mixin import HaloMixin
from distdl.nn.padnd import PadNd
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


class DistributedConv1d(torch.nn.Module, HaloMixin, ConvMixin):

    def __init__(self, x_in_sizes, P_cart, *args, **kwargs):

        super(DistributedConv1d, self).__init__()

        self.x_in_sizes = x_in_sizes
        self.P_cart = P_cart

        if not self.P_cart.active:
            return

        # Do this before checking serial so that the layer works properly
        # in the serial case
        self.conv_layer = torch.nn.Conv1d(*args, **kwargs)

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

        self.broadcast_layer = Broadcast(self.P_wb_cart, self.P_cart)

        self.halo_sizes, self.recv_buffer_sizes, self.send_buffer_sizes, self.needed_ranges = \
            self._compute_exchange_info(self.x_in_sizes,
                                        self.conv_layer.kernel_size,
                                        self.conv_layer.stride,
                                        self.conv_layer.padding,
                                        self.conv_layer.dilation,
                                        self.P_cart)

        self.halo_sizes = self.halo_sizes.astype(int)
        self.needed_ranges = self.needed_ranges.astype(int)

        self.needed_slices = assemble_slices(self.needed_ranges[:, 0], self.needed_ranges[:, 1])

        self.pad_layer = PadNd(self.halo_sizes, value=0, partition=self.P_cart)

        self.local_x_in_sizes_padded = self._compute_local_x_in_sizes_padded(self.x_in_sizes,
                                                                             self.P_cart,
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

        w = self.broadcast_layer(self.weight)
        # self.P_cart.print_sequential(f'{self.P_cart.rank}, {self.weight}')
        # w.requires_grad = self.conv_layer.weight.requires_grad
        self.conv_layer.weight = w

        if self.conv_layer.bias is not None:
            b = self.broadcast_layer(self.bias)
            # b.requires_grad = self.conv_layer.bias.requires_grad
            self.conv_layer.bias = b

        input_padded = self.pad_layer(input)
        input_exchanged = self.halo_layer(input_padded)
        input_needed = input_exchanged[self.needed_slices]
        return self.conv_layer(input_needed)
