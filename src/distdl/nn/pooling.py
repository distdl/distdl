import torch

from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.halo_mixin import HaloMixin
from distdl.nn.module import Module
from distdl.nn.padnd import PadNd
from distdl.utilities.slicing import assemble_slices


class PoolingMixin:

    def _compute_min_input_range(self,
                                 idx,
                                 kernel_size,
                                 stride,
                                 padding,
                                 dilation):

        # incorrect, does not take dilation and padding into account
        return stride * idx + 0

    def _compute_max_input_range(self,
                                 idx,
                                 kernel_size,
                                 stride,
                                 padding,
                                 dilation):

        # incorrect, does not take dilation and padding into account
        return stride * idx + kernel_size - 1


class DistributedPoolBase(Module, HaloMixin, PoolingMixin):

    TorchPoolType = None  # noqa F821

    def __init__(self, P_x, *args, **kwargs):

        super(DistributedPoolBase, self).__init__()

        self.P_x = P_x

        if not self.P_x.active:
            return

        self.pool_layer = self.TorchPoolType(*args, **kwargs)

        # We need the halo shape, and other info, to fully populate the pad,
        # and halo exchange layers.

        self.pad_layer = None

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
        self._input_shape = None
        self._input_requires_grad = None

    def _distdl_module_setup(self, input):

        if not self.P_x.active:
            return

        x_global_shape = self._distdl_backend.compute_global_tensor_shape(input[0],
                                                                          self.P_x)
        self.x_global_shape = x_global_shape

        exchange_info = self._compute_exchange_info(x_global_shape,
                                                    self.pool_layer.kernel_size,
                                                    self.pool_layer.stride,
                                                    self.pool_layer.padding,
                                                    [1],  # torch pooling layers have no dilation
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

        # We have to select out the "unused" entries.
        self.needed_slices = assemble_slices(needed_ranges[:, 0],
                                             needed_ranges[:, 1])

        self._distdl_is_setup = True
        self._input_shape = input[0].shape
        self._input_requires_grad = input[0].requires_grad

    def _distdl_module_teardown(self, input):

        # Reset all sub_layers
        self.pad_layer = None
        self.needed_slices = None
        self.halo_layer = None

        self.x_global_shape = None

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
