import numpy as np
import torch

from distdl.nn.broadcast import Broadcast
from distdl.nn.mixins.conv_mixin import ConvMixin
from distdl.nn.module import Module
from distdl.nn.sum_reduce import SumReduce
from distdl.utilities.slicing import compute_subshape


class DistributedChannelConvBase(Module, ConvMixin):

    TorchConvType = None

    def __init__(self, P_x, P_y, P_w,
                 in_channels=1, out_channels=1,
                 bias=True,
                 *args, **kwargs):

        super(DistributedChannelConvBase, self).__init__()

        # P_x is 1    x P_ci x 1 x ... x 1
        self.P_x = P_x
        # P_y is 1    x P_co x 1 x ... x 1
        self.P_y = P_y
        # P_w is P_co x P_ci x 1 x ... x 1
        # Or starts as P_co x P_ci and cast to the above?
        self.P_w = P_w

        self.P_union = self._distdl_backend.Partition()
        if not (self.P_x.active or
                self.P_y.active or
                self.P_w.active):
            return

        # This guarantees that P_union rank 0 has the kernel size, stride,
        # padding, and dilation factors
        P_union = P_w.create_partition_union(P_x)
        P_union = P_union.create_partition_union(P_y)
        self.P_union = P_union

        P_w_shape = None
        if P_union.rank == 0:
            P_w_shape = np.array(P_w.shape, dtype=np.int)
        P_w_shape = P_union.broadcast_data(P_w_shape, root=0)

        P_co = P_w_shape[0]
        P_ci = P_w_shape[1]
        P_channels = [P_co, P_ci]

        P_x_new_shape = []
        if self.P_x.active:
            if(np.any(P_x.shape[2:] != P_w_shape[2:])):
                raise ValueError("Spatial components of P_x and P_w must match.")
            if(np.any(P_x.shape[2:] != np.ones(len(P_x.shape[2:])))):
                raise ValueError("Spatial components of P_x must be 1 x ... x 1.")
            if P_w_shape[1] != P_x.shape[1]:
                raise ValueError("Index 2 of P_w dimension must match input channel partition.")
            P_x_new_shape = list(P_x.shape)
            P_x_new_shape.insert(1, 1)
            # Currently a hack, removing the batch dimension because P_w does
            # not have one. This is OK because we assume there are no partitions
            # in the batch dimension.
            P_x_new_shape = np.asarray(P_x_new_shape[1:], dtype=int)

        # For the purposes of this layer, we re-cast P_x to have the extra
        # dimension.  This has no impact outside of the layer or on the results.
        self.P_x = self.P_x.create_cartesian_topology_partition(P_x_new_shape)

        P_y_new_shape = []
        if self.P_y.active:
            if(np.any(P_y.shape[2:] != P_w_shape[2:])):
                raise ValueError("Spatial components of P_y and P_w must match.")
            if(np.any(P_y.shape[2:] != np.ones(len(P_y.shape[2:])))):
                raise ValueError("Spatial components of P_y must be 1 x ... x 1.")
            if P_w_shape[0] != P_y.shape[1]:
                raise ValueError("Index 1 of P_w dimension must match output channel partition.")
            P_y_new_shape = list(P_y.shape)
            P_y_new_shape.insert(2, 1)
            # Currently a hack, removing the batch dimension because P_w does
            # not have one. This is OK because we assume there are no partitions
            # in the batch dimension.
            P_y_new_shape = np.asarray(P_y_new_shape[1:], dtype=int)

        # For the purposes of this layer, we re-cast P_x to have the extra
        # dimension.  This has no impact outside of the layer or on the results.
        self.P_y = self.P_y.create_cartesian_topology_partition(P_y_new_shape)

        self.serial = False
        if self.P_w.size == 1:
            self.serial = True
            self.conv_layer = self.TorchConvType(*args, **kwargs)
            return

        self.receives_weight = False
        self.receives_bias = False
        self.stores_bias = False

        self.global_bias = bias

        # Determine P_r, initialize weights there
        if self.P_w.active:

            # Let the P_co column store the bias if it is to be used
            self.stores_bias = self.global_bias if (self.P_w.index[1] == 0) else False

            # Correct the input arguments based on local properties
            local_kwargs = {}
            local_kwargs.update(kwargs)

            # Do this before checking serial so that the layer works properly
            # in the serial case
            local_channels = compute_subshape(P_channels, P_w.index[0:2], [out_channels, in_channels])
            local_out_channels, local_in_channels = local_channels
            local_kwargs["in_channels"] = local_in_channels
            local_kwargs["out_channels"] = local_out_channels

            local_kwargs["bias"] = self.stores_bias
            self.conv_layer = self.TorchConvType(*args, **local_kwargs)

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_shape = None
        self._input_requires_grad = None

        self.x_broadcast = Broadcast(self.P_x, self.P_w, preserve_batch=True)
        self.y_sum_reduce = SumReduce(self.P_w, self.P_y, preserve_batch=True)

    def _distdl_module_setup(self, input):

        if not (self.P_x.active or
                self.P_y.active or
                self.P_w.active):
            return

        if self.serial:
            return

        self._distdl_is_setup = True
        self._input_shape = input[0].shape
        self._input_requires_grad = input[0].requires_grad

    def _distdl_module_teardown(self, input):

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

        if not (self.P_x.active or
                self.P_y.active or
                self.P_w.active):
            return input.clone()

        if self.serial:
            return self.conv_layer(input)

        x = self.x_broadcast(input)

        if self.P_w.active:
            x = self.conv_layer(x)

        y = self.y_sum_reduce(x)

        return y


class DistributedChannelConv1d(DistributedChannelConvBase):

    TorchConvType = torch.nn.Conv1d


class DistributedChannelConv2d(DistributedChannelConvBase):

    TorchConvType = torch.nn.Conv2d


class DistributedChannelConv3d(DistributedChannelConvBase):

    TorchConvType = torch.nn.Conv3d
