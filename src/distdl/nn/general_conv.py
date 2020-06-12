import numpy as np
import torch

from distdl.nn.broadcast import Broadcast
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.halo_mixin import HaloMixin
from distdl.nn.module import Module
from distdl.nn.padnd import PadNd
from distdl.nn.sum_reduce import SumReduce
from distdl.nn.unpadnd import UnpadNd
from distdl.utilities.slicing import assemble_slices
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.slicing import range_coords
from distdl.utilities.torch import NoneTensor


class ConvMixin:

    def _compute_min_input_range(self,
                                 idx,
                                 kernel_size,
                                 stride,
                                 padding,
                                 dilation):

        # incorrect, does not take stride and dilation into account
        # padding might also not be correct in these cases...
        kernel_offsets = (kernel_size - 1) / 2

        # for even sized kernels, always shortchange the left side
        kernel_offsets[kernel_size % 2 == 0] -= 1

        bases = idx + kernel_offsets - padding
        return bases - kernel_offsets

    def _compute_max_input_range(self,
                                 idx,
                                 kernel_size,
                                 stride,
                                 padding,
                                 dilation):

        # incorrect, does not take stride and dilation into account
        # padding might also not be correct in these cases...
        kernel_offsets = (kernel_size - 1) / 2

        bases = idx + kernel_offsets - padding
        return bases + kernel_offsets


class DistributedGeneralConvBase(Module, HaloMixin, ConvMixin):

    TorchConvType = None

    def __init__(self, P_x, P_y, P_w,
                 in_channels=1, out_channels=1,
                 bias=True,
                 *args, **kwargs):

        super(DistributedGeneralConvBase, self).__init__()

        # P_x is 1    x P_ci x P_d-1 x ... x P_0
        self.P_x = P_x
        # P_y is 1    x P_co x P_d-1 x ... x P_0
        self.P_y = P_y
        # P_w is P_co x P_ci x P_d-1 x ... x P_0
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

        P_w_dims = None
        if P_union.rank == 0:
            P_w_dims = np.array(P_w.dims, dtype=np.int)
        P_w_dims = P_union.broadcast_data(P_w_dims, root=0)

        P_co = P_w_dims[0]
        P_ci = P_w_dims[1]
        P_channels = [P_co, P_ci]

        P_x_new_dims = []
        if self.P_x.active:
            if(np.any(P_x.dims[2:] != P_w_dims[2:])):
                raise ValueError("Spatial components of P_x and P_w must match.")
            if P_w_dims[1] != P_x.dims[1]:
                raise ValueError("Index 2 of P_w dimension must match input channel partition.")
            P_x_new_dims = list(P_x.dims)
            P_x_new_dims.insert(1, 1)
            # Currently a hack, removing the batch dimension because P_w does
            # not have one. This is OK because we assume there are no partitions
            # in the batch dimension.
            P_x_new_dims = np.asarray(P_x_new_dims[1:], dtype=int)

        # For the purposes of this layer, we re-cast P_x to have the extra
        # dimension.  This has no impact outside of the layer or on the results.
        self.P_x = self.P_x.create_cartesian_topology_partition(P_x_new_dims)

        P_y_new_dims = []
        if self.P_y.active:
            if(np.any(P_y.dims[2:] != P_w_dims[2:])):
                raise ValueError("Spatial components of P_y and P_w must match.")
            if P_w_dims[0] != P_y.dims[1]:
                raise ValueError("Index 1 of P_w dimension must match output channel partition.")
            P_y_new_dims = list(P_y.dims)
            P_y_new_dims.insert(2, 1)
            # Currently a hack, removing the batch dimension because P_w does
            # not have one. This is OK because we assume there are no partitions
            # in the batch dimension.
            P_y_new_dims = np.asarray(P_y_new_dims[1:], dtype=int)

        # For the purposes of this layer, we re-cast P_x to have the extra
        # dimension.  This has no impact outside of the layer or on the results.
        self.P_y = self.P_y.create_cartesian_topology_partition(P_y_new_dims)

        P_spatial = P_w_dims[2:]

        self.serial = False
        if self.P_w.size == 1:
            self.serial = True
            self.conv_layer = self.TorchConvType(*args, **kwargs)
            return

        self.receives_weight = False
        self.stores_weight = False
        self.receives_bias = False
        self.stores_bias = False

        # Determine P_r, initialize weights there
        if self.P_w.active:
            # All of P_w always receives the weight
            self.receives_weight = True

            # This subset is taken to be the origin of the spartial component
            w_root_subset = []
            for i, c in enumerate(range_coords(P_w.dims)):
                c = np.asarray(c)
                # Find the P_co x P_ci x 1 x ... x 1 subset to store the weights
                if np.all(c[2:] == 0):
                    w_root_subset.append(i)

            self.P_wr_base = self.P_w.create_partition_inclusive(w_root_subset)
            # ones are needed so the broadcast will work
            self.P_wr = self.P_wr_base.create_cartesian_topology_partition([P_co, P_ci] + [1]*len(P_spatial))
            self.stores_weight = self.P_wr.active

            b_subset = []
            for i, c in enumerate(range_coords(P_w.dims)):
                c = np.asarray(c)
                # Find the P_co x 1 x P_0 x ... x P_D-1 subset that needs biases in its calculation.
                # This is everywhere that the input channels is rank 0.
                if c[1] == 0:
                    b_subset.append(i)

            self.P_b_base = self.P_w.create_partition_inclusive(b_subset)
            self.P_b = self.P_b_base.create_cartesian_topology_partition([P_co] + [1] + list(P_spatial))
            self.receives_bias = self.P_b.active and bias

            # Now find the subset of _that_ which actually stores the learnable parameter.
            b_root_subset = []
            for i, c in enumerate(range_coords(P_w.dims)):
                c = np.asarray(c)
            # Find the P_co x 1 x 1 x ... x 1 subset to store the biases
                if np.all(c[1:] == 0):
                    b_root_subset.append(i)

            self.P_br_base = self.P_w.create_partition_inclusive(b_root_subset)
            # ones are needed so the broadcast will work
            self.P_br = self.P_br_base.create_cartesian_topology_partition([P_co] + [1] + [1]*len(P_spatial))
            self.stores_bias = self.P_br.active and bias

            # Correct the input arguments based on local properties
            local_kwargs = {}
            local_kwargs.update(kwargs)

            # Do this before checking serial so that the layer works properly
            # in the serial case
            local_channels = compute_subshape(P_channels, P_w.coords[0:2], [out_channels, in_channels])
            local_out_channels, local_in_channels = local_channels
            local_kwargs["in_channels"] = local_in_channels
            local_kwargs["out_channels"] = local_out_channels

            local_kwargs["bias"] = self.receives_bias
            self.conv_layer = self.TorchConvType(*args, **local_kwargs)

            # If we store the weight it is a learnable parameter iff it is
            # learnable by default in the layer, which it is.
            if self.stores_weight:
                self._weight = torch.nn.Parameter(self.conv_layer.weight.detach())
            else:
                self._weight = NoneTensor()
            # This always exists so we can copy the property
            self._weight.requires_grad = self.conv_layer.weight.requires_grad

            # https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677/2
            new_weight = self.conv_layer.weight.detach() * 0
            new_weight.requires_grad = self.conv_layer.weight.requires_grad
            del self.conv_layer.weight
            self.conv_layer.weight = new_weight

            # If we store the bias, it is a learnable parameter iff it is
            # learnable by default in the layer, which is only true if it
            # exists.
            if self.stores_bias:
                self._bias = torch.nn.Parameter(self.conv_layer.bias.detach())
            else:
                self._bias = NoneTensor()
            # This does not always exist, but when it does we can copy the
            # property.
            if self.receives_bias:
                self._bias.requires_grad = self.conv_layer.bias.requires_grad

                # https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677/2
                new_bias = self.conv_layer.bias.detach() * 0
                new_bias.requires_grad = self.conv_layer.bias.requires_grad
                del self.conv_layer.bias
                self.conv_layer.bias = new_bias

        # Now we need to share the kernel structure.  The size of the kernel
        # is always the spatial dimensions.
        self.conv_kernel_size = None
        self.conv_stride = None
        self.conv_padding = None
        self.conv_dilation = None
        if P_union.rank == 0:
            self.conv_kernel_size = np.array(self.conv_layer.kernel_size, dtype=np.int)
            self.conv_stride = np.array(self.conv_layer.stride, dtype=np.int)
            self.conv_padding = np.array(self.conv_layer.padding, dtype=np.int)
            self.conv_dilation = np.array(self.conv_layer.dilation, dtype=np.int)
        self.conv_kernel_size = P_union.broadcast_data(self.conv_kernel_size, root=0)
        self.conv_stride = P_union.broadcast_data(self.conv_stride, root=0)
        self.conv_padding = P_union.broadcast_data(self.conv_padding, root=0)
        self.conv_dilation = P_union.broadcast_data(self.conv_dilation, root=0)

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

        if P_w.active:
            self.w_broadcast = Broadcast(self.P_wr, self.P_w)

        if self.receives_bias or self.stores_bias:
            self.b_broadcast = Broadcast(self.P_br, self.P_b)

        self.x_broadcast = Broadcast(self.P_x, self.P_w)
        self.y_sum_reduce = SumReduce(self.P_w, self.P_y)

    def _distdl_module_setup(self, input):

        if not (self.P_x.active or
                self.P_y.active or
                self.P_w.active):
            return

        if self.serial:
            return

        x_global_shape = self._distdl_backend.compute_global_tensor_shape(input[0],
                                                                          self.P_x,
                                                                          self.P_union)
        if self.P_x.active:
            exchange_info = self._compute_exchange_info(x_global_shape,
                                                        self.conv_kernel_size,
                                                        self.conv_stride,
                                                        self.conv_padding,
                                                        self.conv_dilation,
                                                        self.P_x.active,
                                                        self.P_x.dims,
                                                        self.P_x.coords)
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

        # The output has to do some unpadding
        if self.P_y.active:

            # This is safe because there are never halos on the channel or batch
            # dimensions.  Therefore, because we assume that the spatial partition
            # of P_x and P_y is the same, then the halo sizes this will
            # compute will also be the same.
            exchange_info = self._compute_exchange_info(x_global_shape,
                                                        self.conv_kernel_size,
                                                        self.conv_stride,
                                                        self.conv_padding,
                                                        self.conv_dilation,
                                                        self.P_y.active,
                                                        self.P_y.dims,
                                                        self.P_y.coords)
            y_halo_shape = exchange_info[0]

            # Unpad sizes are padding in the dimensions where we have a halo,
            # otherwise 0
            conv_padding = np.concatenate(([0, 0], self.conv_padding))
            unpad_shape = []
            for pad, halo in zip(conv_padding, y_halo_shape):
                unpad_shape.append(np.where(halo > 0, pad, 0))
            unpad_shape = np.asarray(unpad_shape)

            self.unpad_layer = UnpadNd(unpad_shape, value=0)

        self._distdl_is_setup = True
        self._input_shape = input[0].shape
        self._input_requires_grad = input[0].requires_grad

    def _distdl_module_teardown(self, input):

        # Reset all sub_layers
        self.pad_layer = None
        self.unpad_layer = None
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

        if not (self.P_x.active or
                self.P_y.active or
                self.P_w.active):
            return input.clone()

        if self.serial:
            return self.conv_layer(input)

        x = input
        if self.P_x.active:
            x = self.pad_layer(x)
            x = self.halo_layer(x)
            x = x[self.needed_slices]

        # Weights always received
        if self.P_w.active:
            w = self.w_broadcast(self._weight)
            self.conv_layer.weight = w

        # Biases only received in some places
        if self.receives_bias or self.stores_bias:
            b = self.b_broadcast(self._bias)
            self.conv_layer.bias = b

        x = self.x_broadcast(x)

        if self.P_w.active:
            x = self.conv_layer(x)

        y = self.y_sum_reduce(x)

        if self.P_y.active:
            y = self.unpad_layer(y)

        return y


class DistributedGeneralConv1d(DistributedGeneralConvBase):

    TorchConvType = torch.nn.Conv1d


class DistributedGeneralConv2d(DistributedGeneralConvBase):

    TorchConvType = torch.nn.Conv2d


class DistributedGeneralConv3d(DistributedGeneralConvBase):

    TorchConvType = torch.nn.Conv3d
