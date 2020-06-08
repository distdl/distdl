import torch

from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.halo_mixin import HaloMixin
from distdl.nn.padnd import PadNd
from distdl.utilities.slicing import assemble_slices


class PoolingMixin:

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


class DistributedPoolBase(torch.nn.Module, HaloMixin, PoolingMixin):

    TorchPoolType = None  # noqa F821

    def __init__(self, x_in_sizes, P_cart, *args, **kwargs):

        super(TorchPoolType, self).__init__()

        self.x_in_sizes = x_in_sizes
        self.P_cart = P_cart

        if not self.P_cart.active:
            return

        self.pool_layer = torch.nn.TorchPoolType(*args, **kwargs)

        self.halo_sizes, self.recv_buffer_sizes, self.send_buffer_sizes, self.needed_ranges = \
            self._compute_exchange_info(self.x_in_sizes,
                                        self.pool_layer.kernel_size,
                                        self.pool_layer.stride,
                                        self.pool_layer.padding,
                                        [1],  # torch pooling layers have no dilation
                                        self.P_cart.active,
                                        self.P_cart.dims,
                                        self.P_cart.coords)

        self.needed_slices = assemble_slices(self.needed_ranges[:, 0], self.needed_ranges[:, 1])

        self.pad_layer = PadNd(self.halo_sizes, value=0, partition=self.P_cart)

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

        input_padded = self.pad_layer(input)
        input_exchanged = self.halo_layer(input_padded)
        input_needed = input_exchanged[self.needed_slices]
        return self.pool_layer(input_needed)


class DistributedAvgPool1d(DistributedPoolBase):

    TorchPoolType = torch.nn.AvgPool1d


class DistributedAvgPool2d(DistributedPoolBase):

    TorchPoolType = torch.nn.AvgPool2d


class DistributedAvgPool3d(DistributedPoolBase):

    TorchPoolType = torch.nn.AvgPool3d


class DistributedMaxPool1d(DistributedPoolBase):

    TorchPoolType = torch.nn.MaxPool1d


class DistributedMaxPool2d(DistributedPoolBase):

    TorchPoolType = torch.nn.MaxPool2d


class DistributedMaxPool3d(DistributedPoolBase):

    TorchPoolType = torch.nn.MaxPool3d
