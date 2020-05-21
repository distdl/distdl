import numpy as np
import torch
from mpi4py import MPI

from distdl.utilities.slicing import compute_nd_slice_volume
from distdl.utilities.slicing import compute_partition_intersection
from distdl.utilities.slicing import compute_subsizes
from distdl.utilities.slicing import range_coords


class DistributedTransposeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_common, sizes,
                P_in, in_data, in_buffers,
                P_out, out_data, out_buffers):

        ctx.P_common = P_common
        ctx.sizes = sizes

        ctx.P_in = P_in
        ctx.in_data = in_data
        ctx.in_buffers = in_buffers

        ctx.P_out = P_out
        ctx.out_data = out_data
        ctx.out_buffers = out_buffers

        if P_common.size == 1:
            return input.clone()

        requests = []

        # If I am getting data, recv my output parts
        recv_count = 0
        if P_out.active:
            for (sl, sz, partner), buff in zip(out_data, out_buffers):
                if buff is not None:
                    req = P_common.comm.Irecv(buff, source=partner, tag=111)
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
                    req = P_common.comm.Isend(buff, dest=partner, tag=111)
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
            output = np.zeros(out_sizes, dtype=np.float64)

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
            return torch.from_numpy(output)
        else:
            return None

    @staticmethod
    def backward(ctx, grad_output):

        P_common = ctx.P_common
        sizes = ctx.sizes

        P_in = ctx.P_in
        in_data = ctx.in_data
        in_buffers = ctx.in_buffers

        P_out = ctx.P_out
        out_data = ctx.out_data
        out_buffers = ctx.out_buffers

        if P_common.size == 1:
            return grad_output.clone(), None, None, None, None, None, None, None, None

        requests = []

        # Recv my input parts
        recv_count = 0
        if P_in.active:
            for (sl, sz, partner), buff in zip(in_data, in_buffers):
                if buff is not None:
                    req = P_common.comm.Irecv(buff, source=partner, tag=113)
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
                    req = P_common.comm.Isend(buff, dest=partner, tag=113)
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
            grad_input = np.zeros(in_sizes, dtype=np.float64)

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
            return torch.from_numpy(grad_input), None, None, None, None, None, None, None, None
        else:
            return None, None, None, None, None, None, None, None, None


class DistributedTranspose(torch.nn.Module):

    def __init__(self, sizes, P_in, P_out):
        super(DistributedTranspose, self).__init__()

        self.sizes = sizes
        self.P_in = P_in
        self.P_out = P_out

        # Find the common partition between the input and output partitions
        P_common = P_in.common_ancestor(P_out)
        if P_common is None:
            raise Exception()
        self.P_common = P_common

        self.in_data = []
        self.out_data = []

        in_dims = P_in.dims
        out_dims = P_out.dims

        # We need to move data between two potentially disjoint partitions,
        # so we have to find a common mapping, which requires the common
        # partition.
        P_common_to_P_in = P_in.map_from_ancestor(P_common)
        P_common_to_P_out = P_out.map_from_ancestor(P_common)

        # We only need to move data to the output partition if we actually
        # have input data.  It is possible to have both input and output data,
        # either input or output data, or neither.  Hence the active guard.
        if P_in.active:
            in_coords = P_in.cartesian_coordinates(P_in.rank)

            # Compute our overlaps for each output subpartition.
            for rank, out_coords in enumerate(range_coords(P_out.dims)):
                sl = compute_partition_intersection(in_dims, in_coords,
                                                    out_dims, out_coords,
                                                    sizes)
                if sl is not None:
                    sz = compute_nd_slice_volume(sl)
                    # Reverse the mapping to get the output partner's rank in
                    # the common partition.
                    partner = np.where(P_common_to_P_out == rank)[0][0]
                    self.in_data.append((sl, sz, partner))
                else:
                    self.in_data.append((None, None, P_out.null_rank))

        # We only need to obtain data from the input partition if we actually
        # have output data.
        if P_out.active:
            out_coords = P_out.cartesian_coordinates(P_out.rank)

            # Compute our overlaps for each input subpartition.
            for rank, in_coords in enumerate(range_coords(P_in.dims)):
                sl = compute_partition_intersection(out_dims, out_coords,
                                                    in_dims, in_coords,
                                                    sizes)
                if sl is not None:
                    sz = compute_nd_slice_volume(sl)
                    # Reverse the mapping to get the input partner's rank in
                    # the common partition.
                    partner = np.where(P_common_to_P_in == rank)[0][0]
                    self.out_data.append((sl, sz, partner))
                else:
                    self.out_data.append((None, None, P_in.null_rank))

        # In the sequential case, don't allocate anything, we don't use them.
        if P_in == P_out:
            self.in_buffers = None
            self.out_buffers = None
            self.serial = True
        else:
            # TODO(#25): The dtype should not be fixed, but correcting this is
            #            a thing that needs to be resolved globally.
            buffs = self._allocate_buffers(np.float64)
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

        if not self.serial:
            DistributedTransposeFunction.apply(input, self.P_common, self.sizes,
                                               self.P_in, self.in_data, self.in_buffers,
                                               self.P_out, self.out_data, self.out_buffers)
        else:
            return input.clone()
