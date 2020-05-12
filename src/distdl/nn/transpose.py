import numpy as np
import torch
from mpi4py import MPI

from distdl.utilities.slicing import compute_nd_slice_volume
from distdl.utilities.slicing import compute_partition_intersection
from distdl.utilities.slicing import compute_subsizes


class DistributedTransposeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, parent_comm, sizes,
                in_slices, in_buffers, in_comm,
                out_slices, out_buffers, out_comm):

        ctx.parent_comm = parent_comm
        ctx.sizes = sizes

        size = parent_comm.Get_size()

        ctx.in_slices = in_slices
        ctx.in_buffers = in_buffers
        ctx.in_comm = in_comm

        ctx.out_slices = out_slices
        ctx.out_buffers = out_buffers
        ctx.out_comm = out_comm

        input_numpy = input.detach().numpy()

        requests = []

        # Recv my output parts
        for r in range(size):
            buff = out_buffers[r]
            if buff is not None:
                req = parent_comm.Irecv(buff, source=r, tag=0)
                requests.append(req)

        # Pack and send my input parts
        for s in range(size):
            buff = in_buffers[s]
            if buff is not None:
                sl = tuple(in_slices[s])
                np.copyto(buff, input_numpy[sl].ravel())
                req = parent_comm.Isend(buff, dest=s, tag=0)
                requests.append(req)

        MPI.Request.Waitall(requests)

        coords = out_comm.Get_coords(out_comm.Get_rank())
        out_sizes = compute_subsizes(out_comm.dims, coords, sizes)
        output = np.zeros(out_sizes, dtype=input_numpy.dtype)

        # Unpack my output parts
        for r in range(size):
            buff = out_buffers[r]
            if buff is not None:
                sl = tuple(out_slices[r])
                sh = output[sl].shape
                np.copyto(output[sl], buff.reshape(sh))

        return torch.from_numpy(output)

    @staticmethod
    def backward(ctx, grad_output):
        pass


class DistributedTranspose(torch.nn.Module):

    def __init__(self, sizes, parent_comm, in_comm, out_comm):
        super(DistributedTranspose, self).__init__()

        self.sizes = sizes
        self.parent_comm = parent_comm
        self.in_comm = in_comm
        self.out_comm = out_comm

        in_slices = compute_partition_intersection(in_comm, out_comm, sizes)
        in_buffer_sizes = [None if s is None else
                           compute_nd_slice_volume(s) for s in in_slices]

        self.in_slices = in_slices
        self.in_buffer_sizes = in_buffer_sizes

        out_slices = compute_partition_intersection(out_comm, in_comm, sizes)
        out_buffer_sizes = [None if s is None else
                            compute_nd_slice_volume(s) for s in out_slices]

        self.out_slices = out_slices
        self.out_buffer_sizes = out_buffer_sizes

    def _allocate_buffers(self, dtype):

        in_buffers = []
        for length in self.in_buffer_sizes:
            buff = None
            if length is not None:
                buff = np.zeros(length, dtype=dtype)

            in_buffers.append(buff)

        out_buffers = []
        for length in self.out_buffer_sizes:
            buff = None
            if length is not None:
                buff = np.zeros(length, dtype=dtype)

            out_buffers.append(buff)

        return in_buffers, out_buffers

    def forward(self, input):

        # DistributedTransposeFunction.apply(input, self.in_buffers, self.out_buffers)
        pass
