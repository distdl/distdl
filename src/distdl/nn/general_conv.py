import numpy as np
import torch

from distdl.nn.broadcast import Broadcast
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.halo_mixin import HaloMixin
from distdl.nn.padnd import PadNd
from distdl.nn.sum_reduce import SumReduce
from distdl.nn.unpadnd import UnPadNd
from distdl.utilities.slicing import assemble_slices
from distdl.utilities.slicing import compute_subsizes
from distdl.utilities.slicing import range_coords
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


class DistributedGeneralConvBase(torch.nn.Module, HaloMixin, ConvMixin):

    TorchConvType = None

    def __init__(self, x_in_sizes, P_x, P_y, P_w, *args, **kwargs):

        super(DistributedGeneralConvBase, self).__init__()

        self.x_in_sizes = x_in_sizes
        self.P_x = P_x
        self.P_y = P_y
        self.P_w = P_w

        if not (self.P_x.active or
                self.P_y.active or
                self.P_w.active):
            return

        # This guarantees that P_union rank 0 has the kernel size, stride,
        # padding, and dilation factors
        P_union = P_w.create_partition_union(P_x)
        P_union = P_union.create_partition_union(P_y)

        # P_w is P_co x P_ci x P_d-1 x ... x P_0
        # P_x is 1    x P_ci x P_d-1 x ... x P_0
        # P_y is 1    x P_co x P_d-1 x ... x P_0
        if P_union.rank == 0:
            P_w_dims = np.array(P_w.dims, dtype=np.int)
        else:
            if self.P_x.active:
                P_w_dim = P_x.dim
            elif self.P_y.active:
                P_w_dim = P_y.dim
            else:  # P_w.active
                P_w_dim = P_w.dim
            P_w_dims = np.zeros(P_w_dim, dtype=np.int)
        P_union.comm.Bcast(P_w_dims, root=0)

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

        # Determine P_r, initialize weights there
        if self.P_w.active:
            # This subset is taken to be the origin of the spartial component
            w_root_subset = []
            for i, c in enumerate(range_coords(P_w.dims)):
                c = np.asarray(c)
                if np.all(c[2:] == 0):
                    w_root_subset.append(i)

            self.P_wr_base = self.P_w.create_partition_inclusive(w_root_subset)
            # ones are needed so the broadcast will work
            self.P_wr = self.P_wr_base.create_cartesian_topology_partition([P_co, P_ci] + [1]*len(P_spatial))
            self.has_weight = self.P_wr.active

            in_channels = kwargs['in_channels']
            out_channels = kwargs['out_channels']

            local_channels = compute_subsizes(P_channels, P_w.coords[0:2], [out_channels, in_channels])
            local_out_channels, local_in_channels = local_channels

        #     # Ignore the bias for now.
        #     kwargs.update({"bias": False})
        #     # bias = False

            # Do this before checking serial so that the layer works properly
            # in the serial case
            local_kwargs = {}
            local_kwargs.update(kwargs)
            local_kwargs["in_channels"] = local_in_channels
            local_kwargs["out_channels"] = local_out_channels
            self.conv_layer = self.TorchConvType(*args, **local_kwargs)

            if self.has_weight:
                self._weight = torch.nn.Parameter(self.conv_layer.weight.detach())
                # if self.conv_layer.bias is not None:
                #     self.bias = torch.nn.Parameter(self.conv_layer.bias.detach())
                # bias = self.bias if (self.P_wr.coords[-1] == 0) else False
            else:
                self._weight = NoneTensor()
                # if self.conv_layer.bias is not None:
                #     self.bias = NoneTensor()

            self._weight.requires_grad = self.conv_layer.weight.requires_grad

            # if self.conv_layer.bias is not None:
            #     self.bias.requires_grad = self.conv_layer.bias.requires_grad

            # https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677/2
            new_weight = self.conv_layer.weight.detach() * 0
            new_weight.requires_grad = self.conv_layer.weight.requires_grad
            del self.conv_layer.weight
            self.conv_layer.weight = new_weight

            # if self.conv_layer.bias is not None:
            #     new_bias = self.conv_layer.bias.detach() * 0
            #     new_bias.requires_grad = self.conv_layer.bias.requires_grad
            #     del self.conv_layer.bias
            #     self.conv_layer.bias = new_bias

            # P_w.print_sequential(f"{self.conv_layer.weight.shape}")

        # Now we need to share the kernel structure.  The size of the kernel
        # is always the spatial dimensions.
        if P_union.rank == 0:
            self.conv_kernel_size = np.array(self.conv_layer.kernel_size, dtype=np.int)
            self.conv_stride = np.array(self.conv_layer.stride, dtype=np.int)
            self.conv_padding = np.array(self.conv_layer.padding, dtype=np.int)
            self.conv_dilation = np.array(self.conv_layer.dilation, dtype=np.int)
        else:
            self.conv_kernel_size = np.zeros(len(P_spatial), dtype=np.int)
            self.conv_stride = np.zeros(len(P_spatial), dtype=np.int)
            self.conv_padding = np.zeros(len(P_spatial), dtype=np.int)
            self.conv_dilation = np.zeros(len(P_spatial), dtype=np.int)
        P_union.comm.Bcast(self.conv_kernel_size, root=0)
        P_union.comm.Bcast(self.conv_stride, root=0)
        P_union.comm.Bcast(self.conv_padding, root=0)
        P_union.comm.Bcast(self.conv_dilation, root=0)

        if self.P_x.active:
            self.halo_sizes, self.recv_buffer_sizes, self.send_buffer_sizes, self.needed_ranges = \
                self._compute_exchange_info(self.x_in_sizes,
                                            self.conv_kernel_size,
                                            self.conv_stride,
                                            self.conv_padding,
                                            self.conv_dilation,
                                            self.P_x.active,
                                            self.P_x.dims,
                                            self.P_x.coords)

            self.halo_sizes = self.halo_sizes.astype(int)
            self.needed_ranges = self.needed_ranges.astype(int)

            self.pad_layer = PadNd(self.halo_sizes, value=0, partition=self.P_x)

            self.local_x_in_sizes_padded = self._compute_local_x_in_sizes_padded(self.x_in_sizes,
                                                                                 self.P_x.dims,
                                                                                 self.P_x.coords,
                                                                                 self.halo_sizes)
            self.halo_layer = HaloExchange(self.local_x_in_sizes_padded,
                                           self.halo_sizes,
                                           self.recv_buffer_sizes,
                                           self.send_buffer_sizes,
                                           self.P_x)
            self.needed_slices = assemble_slices(self.needed_ranges[:, 0], self.needed_ranges[:, 1])

        # The output has to do some unpadding
        if self.P_y.active:

            # This is safe because there are never halos on the channel or batch
            # dimensions.  Therefore, because we assume that the spatial partition
            # of P_x and P_y is the same, then the halo sizes this will
            # compute will also be the same.  Note, we will not overwrite
            # any variables if we are also P_x.
            y_halo_sizes, ignore1, ignore2, y_needed_ranges = \
                self._compute_exchange_info(self.x_in_sizes,
                                            self.conv_kernel_size,
                                            self.conv_stride,
                                            self.conv_padding,
                                            self.conv_dilation,
                                            self.P_y.active,
                                            self.P_y.dims,
                                            self.P_y.coords)

            # Unpad sizes are padding in the dimensions where we have a halo,
            # otherwise 0
            self.pads = np.concatenate(([0, 0], self.conv_padding))
            self.unpad_sizes = []

            for pad, halo_size in zip(self.pads, y_halo_sizes):
                self.unpad_sizes.append(np.where(halo_size > 0, pad, 0))
            self.unpad_sizes = np.asarray(self.unpad_sizes)

            self.unpad_layer = UnPadNd(self.unpad_sizes, value=0, partition=self.P_y)

        if P_w.active:
            self.w_broadcast = Broadcast(self.P_wr, self.P_w)

        self.x_broadcast = Broadcast(self.P_x, self.P_w)
        self.y_sum_reduce = SumReduce(self.P_w, self.P_y)

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

        if self.P_w.active:
            w = self.w_broadcast(self._weight)
            self.conv_layer.weight = w

        x = self.x_broadcast(x)

        if self.P_w.active:
            x = self.conv_layer(x)

        y = self.y_sum_reduce(x)

        if self.P_y.active:
            y = self.unpad_layer(y)

        return y

        # if self.conv_layer.bias is not None:
        #     b = self.broadcast_layer(self.bias)
        #     self.conv_layer.bias = b


class DistributedGeneralConv1d(DistributedGeneralConvBase):

    TorchConvType = torch.nn.Conv1d


class DistributedGeneralConv2d(DistributedGeneralConvBase):

    TorchConvType = torch.nn.Conv2d


class DistributedGeneralConv3d(DistributedGeneralConvBase):

    TorchConvType = torch.nn.Conv3d
