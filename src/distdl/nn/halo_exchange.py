import numpy as np

from distdl.nn.module import Module
from distdl.utilities.slicing import compute_nd_slice_volume


class HaloExchange(Module):

    def __init__(self, x_in_sizes, halo_sizes, recv_buffer_sizes, send_buffer_sizes, cartesian_partition):

        super(HaloExchange, self).__init__()

        self.x_in_sizes = x_in_sizes
        self.halo_sizes = halo_sizes
        self.recv_buffer_sizes = recv_buffer_sizes
        self.send_buffer_sizes = send_buffer_sizes
        self.cartesian_partition = cartesian_partition

        if cartesian_partition.active:
            self.neighbor_ranks = self.cartesian_partition.neighbor_ranks(self.cartesian_partition.rank)

            self.slices = self._assemble_slices(self.x_in_sizes, self.recv_buffer_sizes, self.send_buffer_sizes)
            self.buffers = self._allocate_buffers(self.slices, self.recv_buffer_sizes, self.send_buffer_sizes)
        else:
            self.neighbor_ranks = None
            self.slices = None
            self.buffers = None

    def _assemble_slices(self, x_in_sizes, recv_buffer_sizes, send_buffer_sizes):

        dim = len(x_in_sizes)

        slices = []

        for i in range(dim):
            slices_i = [[], [], [], []]

            for j in range(dim):
                s = x_in_sizes[j]

                lrecv_size = int(recv_buffer_sizes[j, 0])
                lsend_size = int(send_buffer_sizes[j, 0])
                rrecv_size = int(recv_buffer_sizes[j, 1])
                rsend_size = int(send_buffer_sizes[j, 1])

                # Left bulk and ghost start/stop values
                lb_start = lrecv_size
                lb_stop = lrecv_size + lsend_size
                lg_start = 0
                lg_stop = lrecv_size

                # Right bulk and ghost start/stop values
                rb_start = s - (rrecv_size + rsend_size)
                rb_stop = s - rrecv_size
                rg_start = s - rrecv_size
                rg_stop = s

                # For each dimension i, we need to define a dim-dimensional
                # rectangular prism that forms the region we are slicing for
                # the left bulk, left ghost, right bulk, and right ghost. These
                # regions are computed in a nested manner than ensures that the
                # values in the corners are correct.

                # When the dimension of the current side of the rectangular
                # prism is less than the dimension of interest for the slices,
                # we take the entire width of the tensor.
                if j < i:
                    slices_i[0].append(slice(lg_start, rg_stop, None))
                    slices_i[1].append(slice(lg_start, rg_stop, None))
                    slices_i[2].append(slice(lg_start, rg_stop, None))
                    slices_i[3].append(slice(lg_start, rg_stop, None))

                # When the dimension of the current side of the rectangular
                # prism is equal to the dimension of interest for the slices,
                # we take distinct values for the width of each bulk and ghost region.
                elif j == i:
                    slices_i[0].append(slice(lb_start, lb_stop, None))
                    slices_i[1].append(slice(lg_start, lg_stop, None))
                    slices_i[2].append(slice(rb_start, rb_stop, None))
                    slices_i[3].append(slice(rg_start, rg_stop, None))

                # Otherwise, we include only the values in the bulk.
                else:
                    slices_i[0].append(slice(lb_start, rb_stop, None))
                    slices_i[1].append(slice(lb_start, rb_stop, None))
                    slices_i[2].append(slice(lb_start, rb_stop, None))
                    slices_i[3].append(slice(lb_start, rb_stop, None))

            slices.append([tuple(x) for x in slices_i])

        return slices

    def _allocate_buffers(self, slices, recv_buffer_sizes, send_buffer_sizes):

        dim = len(slices)

        buffers = []

        for i in range(dim):
            lbb_len = compute_nd_slice_volume(slices[i][0]) if send_buffer_sizes[i, 0] > 0 else 0
            lgb_len = compute_nd_slice_volume(slices[i][1]) if recv_buffer_sizes[i, 0] > 0 else 0
            rbb_len = compute_nd_slice_volume(slices[i][2]) if send_buffer_sizes[i, 1] > 0 else 0
            rgb_len = compute_nd_slice_volume(slices[i][3]) if recv_buffer_sizes[i, 1] > 0 else 0

            buffers_i = [np.zeros(shape=x) if x > 0 else None for x in [lbb_len, lgb_len, rbb_len, rgb_len]]
            buffers.append(buffers_i)

        return buffers

    def forward(self, input):

        Function = self._distdl_backend.autograd.halo_exchange.HaloExchangeFunction

        return Function.apply(input,
                              self.slices,
                              self.buffers,
                              self.neighbor_ranks,
                              self.cartesian_partition)
