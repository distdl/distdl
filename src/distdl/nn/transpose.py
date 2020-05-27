import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.utilities.slicing import compute_nd_slice_volume
from distdl.utilities.slicing import compute_partition_intersection
from distdl.utilities.slicing import compute_subsizes
from distdl.utilities.slicing import range_coords
from distdl.utilities.torch import NoneTensor


class DistributedTransposeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_union, sizes,
                P_in, in_data, in_buffers,
                P_out, out_data, out_buffers, dtype):

        ctx.P_union = P_union
        ctx.sizes = sizes

        ctx.P_in = P_in
        ctx.in_data = in_data
        ctx.in_buffers = in_buffers

        ctx.P_out = P_out
        ctx.out_data = out_data
        ctx.out_buffers = out_buffers

        ctx.dtype = dtype

        input_requires_grad = False
        # By design, P_in is always first in the union
        if P_union.active:
            if P_in.rank == 0:
                input_requires_grad = input.requires_grad
                P_union.comm.Bcast(np.array([1 if input_requires_grad else 0]),
                                   root=0)
            else:
                irg = np.array([0], dtype=np.int)
                P_union.comm.Bcast(irg, root=0)
                input_requires_grad = bool(irg[0] == 1)

        ctx.input_requires_grad = input_requires_grad

        requests = []

        # Default everyone to output nothing
        output = NoneTensor()

        # If I am getting data, recv my output parts
        recv_count = 0
        if P_out.active:
            for (sl, sz, partner), buff in zip(out_data, out_buffers):
                if buff is not None:
                    req = P_union.comm.Irecv(buff, source=partner, tag=111)
                    requests.append(req)
                else:
                    # We add this if there is no recv so that the indices of
                    # the requests array match the indices of out_data and
                    # out_buffers.
                    requests.append(MPI.REQUEST_NULL)
                recv_count += 1

        # If I have data to share, pack and send my input parts
        send_count = 0
        if P_in.active:
            input_numpy = input.detach().numpy()
            for (sl, sz, partner), buff in zip(in_data, in_buffers):
                if buff is not None:
                    np.copyto(buff, input_numpy[tuple(sl)].ravel())
                    req = P_union.comm.Isend(buff, dest=partner, tag=111)
                    requests.append(req)
                else:
                    # We add this for symmetry, but don't really need it.
                    requests.append(MPI.REQUEST_NULL)
                send_count += 1

        # We do this after the sends so that they can get started before local
        # allocations.
        if P_out.active:
            coords = P_out.cartesian_coordinates(P_out.rank)
            out_sizes = compute_subsizes(P_out.comm.dims, coords, sizes)
            # TODO(#25): The dtype should not be fixed, but correcting this is
            #            a thing that needs to be resolved globally.
            output = np.zeros(out_sizes, dtype=dtype)

        # Unpack the received data as it arrives
        completed_count = 0
        while(completed_count < len(requests)):
            status = MPI.Status()
            index = MPI.Request.Waitany(requests, status)

            # In MPI, we don't get the index out if the request is an
            # instance of MPI.REQUEST_NULL, instead MPI.UNDEFINED is returned.
            if P_out.active and index < recv_count and index != MPI.UNDEFINED:
                # Unpack my output parts
                sl, sz, partner = out_data[index]
                buff = out_buffers[index]
                if buff is not None:
                    sh = output[tuple(sl)].shape
                    np.copyto(output[tuple(sl)], buff.reshape(sh))

            completed_count += 1

        if P_out.active:
            output = torch.from_numpy(output)
            output.requires_grad = input_requires_grad

        return output

    @staticmethod
    def backward(ctx, grad_output):

        P_union = ctx.P_union
        sizes = ctx.sizes

        P_in = ctx.P_in
        in_data = ctx.in_data
        in_buffers = ctx.in_buffers

        P_out = ctx.P_out
        out_data = ctx.out_data
        out_buffers = ctx.out_buffers

        dtype = ctx.dtype

        input_requires_grad = ctx.input_requires_grad

        requests = []

        # Default everyone to output None
        grad_input = NoneTensor()

        # Recv my input parts
        recv_count = 0
        if P_in.active:
            for (sl, sz, partner), buff in zip(in_data, in_buffers):
                if buff is not None:
                    req = P_union.comm.Irecv(buff, source=partner, tag=113)
                    requests.append(req)
                else:
                    # We add this if there is no recv so that the indices of
                    # the requests array match the indices of in_data and
                    # in_buffers.
                    requests.append(MPI.REQUEST_NULL)
                recv_count += 1

        # Pack and send my input parts
        send_count = 0
        if P_out.active:
            grad_output_numpy = grad_output.detach().numpy()
            for (sl, sz, partner), buff in zip(out_data, out_buffers):
                if buff is not None:
                    np.copyto(buff, grad_output_numpy[tuple(sl)].ravel())
                    req = P_union.comm.Isend(buff, dest=partner, tag=113)
                    requests.append(req)
                else:
                    # We add this for symmetry, but don't really need it.
                    requests.append(MPI.REQUEST_NULL)
                send_count += 1

        if P_in.active:
            coords = P_in.cartesian_coordinates(P_in.rank)
            in_sizes = compute_subsizes(P_in.comm.dims, coords, sizes)
            # TODO(#25): The dtype should not be fixed, but correcting this is
            #            a thing that needs to be resolved globally.
            grad_input = np.zeros(in_sizes, dtype=dtype)

        # Unpack the received data as it arrives
        completed_count = 0
        while(completed_count < len(requests)):
            status = MPI.Status()
            index = MPI.Request.Waitany(requests, status)

            # In MPI, we don't get the index out if the request is an
            # instance of MPI.REQUEST_NULL, instead MPI.UNDEFINED is returned.
            if P_in.active and index < recv_count and index != MPI.UNDEFINED:
                # Unpack my output parts
                sl, sz, partner = in_data[index]
                buff = in_buffers[index]
                if buff is not None:
                    sh = grad_input[tuple(sl)].shape
                    # This would normally be an add into the grad_input tensor
                    # but we just created it, so a copy is sufficient.
                    np.copyto(grad_input[tuple(sl)], buff.reshape(sh))

            completed_count += 1

        if P_in.active:
            grad_input = torch.from_numpy(grad_input)
            grad_input.requires_grad = input_requires_grad

        return grad_input, None, None, None, None, None, None, None, None, None


class DistributedTranspose(torch.nn.Module):

    def __init__(self, sizes, P_in, P_out):
        super(DistributedTranspose, self).__init__()

        self.sizes = sizes
        self.P_in = P_in
        self.P_out = P_out

        self.in_data = []
        self.out_data = []

        self.in_buffers = None
        self.out_buffers = None

        # TODO(#25): The dtype should not be fixed, but correcting this is
        #            a thing that needs to be resolved globally.
        self.dtype = np.float32

        self.identity = False

        if P_in == P_out:
            self.identity = True
            return

        P_union = MPIPartition(MPI.COMM_NULL)
        if P_in.active or P_out.active:
            P_union = P_in.create_partition_union(P_out)
        self.P_union = P_union

        if not P_union.active:
            # This is where the early exit stuff will go
            return

        # Find the rank in P_union with rank 0 of P_in
        rank_map_data = np.array([-1], dtype=np.int)
        if P_in.active:
            rank_map_data[0] = P_in.rank
        rank_map = -1*np.ones(P_union.size, dtype=np.int)
        P_union.comm.Allgather(rank_map_data, rank_map)
        in_root = np.where(rank_map == 0)[0][0]

        # Find the rank in P_union with rank 0 of P_dest
        rank_map_data = np.array([-1], dtype=np.int)
        if P_out.active:
            rank_map_data[0] = P_out.rank
        rank_map = -1*np.ones(P_union.size, dtype=np.int)
        P_union.comm.Allgather(rank_map_data, rank_map)
        out_root = np.where(rank_map == 0)[0][0]

        # Share the in cartesian dimension with everyone
        in_dim = np.zeros(1, dtype=np.int)
        if P_in.active and P_in.rank == 0:
            in_dim[0] = P_in.dim
        P_union.comm.Bcast(in_dim, root=in_root)

        # Share the out cartesian dimension with everyone
        out_dim = np.zeros(1, dtype=np.int)
        if P_out.active and P_out.rank == 0:
            out_dim[0] = P_out.dim
        P_union.comm.Bcast(out_dim, root=out_root)

        in_dims = np.ones(in_dim, dtype=np.int)
        if P_in.active and P_in.rank == 0:
            in_dims = P_in.dims
        P_union.comm.Bcast(in_dims, root=in_root)

        # Share the out partition dimensions with everyone
        out_dims = np.zeros(out_dim, dtype=np.int)
        if P_out.active and P_out.rank == 0:
            out_dims = P_out.dims
        P_union.comm.Bcast(out_dims, root=out_root)

        in_index = -1
        if P_in.active:
            in_index = P_in.rank
        out_index = -1
        if P_out.active:
            out_index = P_out.rank

        # Share the two indices with every worker in the union.  The first
        # column of data contains the source index and the second contains
        # the destination index.
        union_indices = -1*np.ones(2*self.P_union.size, dtype=np.int)
        local_indices = np.array([in_index, out_index], dtype=np.int)
        self.P_union.comm.Allgather(local_indices, union_indices)
        union_indices.shape = (-1, 2)

        # We only need to move data to the output partition if we actually
        # have input data.  It is possible to have both input and output data,
        # either input or output data, or neither.  Hence the active guard.
        if P_in.active:
            in_coords = P_in.cartesian_coordinates(P_in.rank)

            # Compute our overlaps for each output subpartition.
            for rank, out_coords in enumerate(range_coords(out_dims)):
                sl = compute_partition_intersection(in_dims, in_coords,
                                                    out_dims, out_coords,
                                                    sizes)
                if sl is not None:
                    sz = compute_nd_slice_volume(sl)
                    # Reverse the mapping to get the output partner's rank in
                    # the common partition.
                    partner = np.where(union_indices[:, 1] == rank)[0][0]
                    self.in_data.append((sl, sz, partner))
                else:
                    self.in_data.append((None, None, None))

        # We only need to obtain data from the input partition if we actually
        # have output data.
        if P_out.active:
            out_coords = P_out.cartesian_coordinates(P_out.rank)

            # Compute our overlaps for each input subpartition.
            for rank, in_coords in enumerate(range_coords(in_dims)):
                sl = compute_partition_intersection(out_dims, out_coords,
                                                    in_dims, in_coords,
                                                    sizes)
                if sl is not None:
                    sz = compute_nd_slice_volume(sl)
                    # Reverse the mapping to get the input partner's rank in
                    # the common partition.
                    partner = np.where(union_indices[:, 0] == rank)[0][0]
                    self.out_data.append((sl, sz, partner))
                else:
                    self.out_data.append((None, None, None))

        buffs = self._allocate_buffers(self.dtype)
        self.in_buffers = buffs[0]
        self.out_buffers = buffs[1]

    def _allocate_buffers(self, dtype):

        in_buffers = []
        for sl, sz, r in self.in_data:
            buff = None
            if sz is not None:
                buff = np.zeros(sz, dtype=dtype)

            in_buffers.append(buff)

        out_buffers = []
        for sl, sz, r in self.out_data:
            buff = None
            if sz is not None:
                buff = np.zeros(sz, dtype=dtype)

            out_buffers.append(buff)

        return in_buffers, out_buffers

    def forward(self, input):

        if self.identity:
            return input.clone()

        return DistributedTransposeFunction.apply(input,
                                                  self.P_union,
                                                  self.sizes,
                                                  self.P_in,
                                                  self.in_data,
                                                  self.in_buffers,
                                                  self.P_out,
                                                  self.out_data,
                                                  self.out_buffers,
                                                  self.dtype)
