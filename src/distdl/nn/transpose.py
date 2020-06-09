import numpy as np
from mpi4py import MPI

from distdl.nn.module import Module
from distdl.utilities.slicing import compute_nd_slice_volume
from distdl.utilities.slicing import compute_partition_intersection
from distdl.utilities.slicing import range_coords


class DistributedTranspose(Module):

    def __init__(self, global_tensor_sizes, P_in, P_out):
        super(DistributedTranspose, self).__init__()

        global_tensor_sizes = np.asarray(global_tensor_sizes)

        self.global_tensor_sizes = global_tensor_sizes
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

        P_union = self._distdl_backend.Partition(MPI.COMM_NULL)
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

        tensor_dim = len(global_tensor_sizes)

        if in_dim != out_dim:
            raise ValueError("Input and output partition must be same dimension.")

        if in_dim != tensor_dim:
            raise ValueError(f"Input partition mush have same dimension "
                             f"({in_dim}) as input tensor rank ({tensor_dim}).")

        if out_dim != tensor_dim:
            raise ValueError(f"Output partition mush have same dimension "
                             f"({out_dim}) as input tensor rank ({tensor_dim}).")

        if 1 in global_tensor_sizes[global_tensor_sizes != out_dims]:
            raise ValueError(f"Input tensor must not be size 1 "
                             f"({global_tensor_sizes}) in a dimension where "
                             f"output partition is other than 1 ({out_dims}).")

        # We only need to move data to the output partition if we actually
        # have input data.  It is possible to have both input and output data,
        # either input or output data, or neither.  Hence the active guard.
        if P_in.active:
            in_coords = P_in.cartesian_coordinates(P_in.rank)

            # Compute our overlaps for each output subpartition.
            for rank, out_coords in enumerate(range_coords(out_dims)):
                sl = compute_partition_intersection(in_dims, in_coords,
                                                    out_dims, out_coords,
                                                    global_tensor_sizes)
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
                                                    global_tensor_sizes)
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

        Function = self._distdl_backend.autograd.transpose.DistributedTransposeFunction

        if self.identity:
            return input.clone()

        return Function.apply(input,
                              self.P_union,
                              self.global_tensor_sizes,
                              self.P_in,
                              self.in_data,
                              self.in_buffers,
                              self.P_out,
                              self.out_data,
                              self.out_buffers,
                              self.dtype)
