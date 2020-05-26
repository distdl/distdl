import numpy as np
import torch

from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.halo_mixin import HaloMixin
from distdl.nn.padnd import PadNd
from distdl.utilities.slicing import assemble_slices
from distdl.utilities.slicing import compute_subsizes


class DistributedAvgPool1d(torch.nn.Module, HaloMixin):

    def __init__(self, x_in_sizes, P_cart, *args, **kwargs):

        super(DistributedAvgPool1d, self).__init__()

        self.x_in_sizes = x_in_sizes
        self.P_cart = P_cart

        self.pool_layer = torch.nn.AvgPool1d(*args, **kwargs)

        self.kernel_sizes = self.pool_layer.kernel_size
        self.strides = self.pool_layer.stride
        self.pads = self.pool_layer.padding
        self.dilations = [0]  # pooling layers do not have dilations

        self.halo_sizes, self.recv_buffer_sizes, self.send_buffer_sizes, self.needed_ranges = \
            self._compute_exchange_info(self.x_in_sizes,
                                        self.kernel_sizes,
                                        self.strides,
                                        self.pads,
                                        self.dilations,
                                        self.P_cart)

        self.needed_slices = None
        if P_cart.active:
            self.needed_slices = tuple(assemble_slices(self.needed_ranges[:, 0], self.needed_ranges[:, 1]))

        self.pad_layer = PadNd(self.halo_sizes, value=0, partition=self.P_cart)

        if P_cart.active:
            self.x_in_sizes_local = compute_subsizes(self.P_cart.dims, self.P_cart.cartesian_coordinates(self.P_cart.rank), self.x_in_sizes)
            self.x_in_sizes_local_padded = [s + lpad + rpad for s, (lpad, rpad) in zip(self.x_in_sizes_local, self.halo_sizes)]
        else:
            self.x_in_sizes_local = None
            self.x_in_sizes_local_padded = None

        self.halo_exchange_layer = HaloExchange(self.x_in_sizes_local_padded,
                                                self.halo_sizes,
                                                self.recv_buffer_sizes,
                                                self.send_buffer_sizes,
                                                self.P_cart)

    def _compute_min_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take dilation and padding into account
        return strides * idx + 0

    def _compute_max_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take dilation and padding into account
        return strides * idx + kernel_sizes - 1

    def forward(self, input):

        if not self.P_cart.active:
            return input.clone()

        input_padded = self.pad_layer(input)
        input_exchanged = self.halo_exchange_layer(input_padded)
        input_needed = input_exchanged[self.needed_slices]
        return self.pool_layer(input_needed)


class DistributedAvgPool2d(torch.nn.Module, HaloMixin):

    def __init__(self, x_in_sizes, P_cart, *args, **kwargs):

        super(DistributedAvgPool2d, self).__init__()

        self.x_in_sizes = x_in_sizes
        self.P_cart = P_cart

        self.pool_layer = torch.nn.AvgPool2d(*args, **kwargs)

        # exchange info expects the kernel sizes and strides to have the dame dimensionality as P_cart.dim
        self.kernel_sizes = np.concatenate((np.array([1, 1]), np.asarray(self.pool_layer.kernel_size)), axis=None)
        self.strides = np.concatenate((np.array([1, 1]), np.asarray(self.pool_layer.stride)), axis=None)

        # Pads and dilations can be broadcast because the are scalars or 1d
        self.pads = self.pool_layer.padding
        self.dilations = [0]  # pooling layers do not have dilations

        self.halo_sizes, self.recv_buffer_sizes, self.send_buffer_sizes, self.needed_ranges = \
            self._compute_exchange_info(self.x_in_sizes,
                                        self.kernel_sizes,
                                        self.strides,
                                        self.pads,
                                        self.dilations,
                                        self.P_cart)

        self.needed_slices = None
        if P_cart.active:
            self.needed_slices = tuple(assemble_slices(self.needed_ranges[:, 0], self.needed_ranges[:, 1]))

        self.pad_layer = PadNd(self.halo_sizes, value=0, partition=self.P_cart)

        if P_cart.active:
            self.x_in_sizes_local = compute_subsizes(self.P_cart.dims, self.P_cart.cartesian_coordinates(self.P_cart.rank), self.x_in_sizes)
            self.x_in_sizes_local_padded = [s + lpad + rpad for s, (lpad, rpad) in zip(self.x_in_sizes_local, self.halo_sizes)]
        else:
            self.x_in_sizes_local = None
            self.x_in_sizes_local_padded = None

        self.halo_exchange_layer = HaloExchange(self.x_in_sizes_local_padded,
                                                self.halo_sizes,
                                                self.recv_buffer_sizes,
                                                self.send_buffer_sizes,
                                                self.P_cart)

    def _compute_min_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take dilation and padding into account
        return strides * idx + 0

    def _compute_max_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take dilation and padding into account
        return strides * idx + kernel_sizes - 1

    def forward(self, input):

        if not self.P_cart.active:
            return input.clone()

        input_padded = self.pad_layer(input)
        input_exchanged = self.halo_exchange_layer(input_padded)
        input_needed = input_exchanged[self.needed_slices]
        return self.pool_layer(input_needed)
