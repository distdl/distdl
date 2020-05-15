import numpy as np
import torch
from mpi4py import MPI

from distdl.utilities.parallel import get_neighbor_ranks
from distdl.utilities.slicing import compute_nd_slice_volume


class HaloExchangeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, slices, buffers, neighbor_ranks, cart_comm):

        ctx.slices = slices
        ctx.buffers = buffers
        ctx.neighbor_ranks = neighbor_ranks
        ctx.cart_comm = cart_comm

        size = cart_comm.Get_size()
        ctx.size = size

        ctx.mark_dirty(input)

        if size == 1:
            return input

        input_numpy = input.detach().numpy()

        dim = cart_comm.dim
        for i in range(dim):

            lbs, lgs, rbs, rgs = slices[i]
            lbb, lgb, rbb, rgb = buffers[i]
            lrank, rrank = neighbor_ranks[i]

            if lbb is not None:
                np.copyto(lbb, input_numpy[lbs].ravel())
            if rbb is not None:
                np.copyto(rbb, input_numpy[rbs].ravel())

            ltag = 0
            rtag = 1

            lrecv_req = cart_comm.Irecv(lgb, source=lrank, tag=rtag) if lgb is not None else MPI.REQUEST_NULL
            lsend_req = cart_comm.Isend(lbb, dest=lrank, tag=ltag) if lbb is not None else MPI.REQUEST_NULL
            rrecv_req = cart_comm.Irecv(rgb, source=rrank, tag=ltag) if rgb is not None else MPI.REQUEST_NULL
            rsend_req = cart_comm.Isend(rbb, dest=rrank, tag=rtag) if rbb is not None else MPI.REQUEST_NULL

            reqs = [lrecv_req, lsend_req, rrecv_req, rsend_req]
            n_reqs_completed = 0

            while n_reqs_completed < len(reqs):
                status = MPI.Status()
                index = MPI.Request.Waitany(reqs, status)

                if index != MPI.UNDEFINED:
                    if index == 0:
                        newshape = input_numpy[lgs].shape
                        np.copyto(input_numpy[lgs], lgb.reshape(newshape))
                    elif index == 2:
                        newshape = input_numpy[rgs].shape
                        np.copyto(input_numpy[rgs], rgb.reshape(newshape))

                n_reqs_completed += 1

        return input

    @staticmethod
    def backward(ctx, grad_output):

        slices = ctx.slices
        buffers = ctx.buffers
        neighbor_ranks = ctx.neighbor_ranks
        cart_comm = ctx.cart_comm
        size = ctx.size

        if size == 1:
            return grad_output

        grad_output_numpy = grad_output.detach().numpy()

        dim = cart_comm.dim
        for i in reversed(range(dim)):

            lbs, lgs, rbs, rgs = slices[i]
            lbb, lgb, rbb, rgb = buffers[i]
            lrank, rrank = neighbor_ranks[i]

            if lgb is not None:
                np.copyto(lgb, grad_output_numpy[lgs].ravel())
                grad_output_numpy[lgs] = 0.0
            if rgb is not None:
                np.copyto(rgb, grad_output_numpy[rgs].ravel())
                grad_output_numpy[rgs] = 0.0

            ltag = 0
            rtag = 1

            lrecv_req = cart_comm.Irecv(lbb, source=lrank, tag=rtag) if lbb is not None else MPI.REQUEST_NULL
            lsend_req = cart_comm.Isend(lgb, dest=lrank, tag=ltag) if lgb is not None else MPI.REQUEST_NULL
            rrecv_req = cart_comm.Irecv(rbb, source=rrank, tag=ltag) if rbb is not None else MPI.REQUEST_NULL
            rsend_req = cart_comm.Isend(rgb, dest=rrank, tag=rtag) if rgb is not None else MPI.REQUEST_NULL

            reqs = [lrecv_req, lsend_req, rrecv_req, rsend_req]
            n_reqs_completed = 0

            while n_reqs_completed < len(reqs):
                status = MPI.Status()
                index = MPI.Request.Waitany(reqs, status)

                if index != MPI.UNDEFINED:
                    if index == 0:
                        newshape = grad_output_numpy[lbs].shape
                        grad_output_numpy[lbs] += lbb.reshape(newshape)
                    elif index == 2:
                        newshape = grad_output_numpy[rbs].shape
                        grad_output_numpy[rbs] += rbb.reshape(newshape)

                n_reqs_completed += 1

        return grad_output, None, None, None, None


class HaloExchange(torch.nn.Module):

    def __init__(self, x_in_sizes, halo_sizes, recv_buffer_sizes, send_buffer_sizes, cart_comm):

        super(HaloExchange, self).__init__()

        self.x_in_sizes = x_in_sizes
        self.halo_sizes = halo_sizes
        self.recv_buffer_sizes = recv_buffer_sizes
        self.send_buffer_sizes = send_buffer_sizes
        self.cart_comm = cart_comm

        self.neighbor_ranks = get_neighbor_ranks(self.cart_comm)

        self.slices = self._assemble_slices(self.x_in_sizes, self.recv_buffer_sizes, self.send_buffer_sizes)
        self.buffers = self._allocate_buffers(self.slices, self.recv_buffer_sizes, self.send_buffer_sizes)

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

                lb_start = lrecv_size
                lb_stop = lrecv_size + lsend_size
                lg_start = 0
                lg_stop = lrecv_size

                rb_start = s - (rrecv_size + rsend_size)
                rb_stop = s - rrecv_size
                rg_start = s - rrecv_size
                rg_stop = s

                if j < i:
                    slices_i[0].append(slice(lg_start, rg_stop, None))
                    slices_i[1].append(slice(lg_start, rg_stop, None))
                    slices_i[2].append(slice(lg_start, rg_stop, None))
                    slices_i[3].append(slice(lg_start, rg_stop, None))

                elif j == i:
                    slices_i[0].append(slice(lb_start, lb_stop, None))
                    slices_i[1].append(slice(lg_start, lg_stop, None))
                    slices_i[2].append(slice(rb_start, rb_stop, None))
                    slices_i[3].append(slice(rg_start, rg_stop, None))

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
        return HaloExchangeFunction.apply(input, self.slices, self.buffers, self.neighbor_ranks, self.cart_comm)
