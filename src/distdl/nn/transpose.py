import numpy as np

from distdl.nn.module import Module
from distdl.utilities.slicing import compute_nd_slice_volume
from distdl.utilities.slicing import compute_partition_intersection
from distdl.utilities.slicing import range_coords


class DistributedTranspose(Module):

    def __init__(self, P_in, P_out):
        super(DistributedTranspose, self).__init__()

        self.global_tensor_sizes = None

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

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_shape = None
        self._input_requires_grad = None

        if P_in == P_out:
            self.identity = True
            return

        P_union = self._distdl_backend.Partition()
        if P_in.active or P_out.active:
            P_union = P_in.create_partition_union(P_out)
        self.P_union = P_union

        self.P_in_dims = None
        self.P_out_dims = None
        self.union_indices = None

        if not P_union.active:
            return

        data = None
        if self.P_in.active:
            data = self.P_in.dims
        self.P_in_dims = self.P_union.broadcast_data(data, P_data=self.P_in)

        data = None
        if self.P_out.active:
            data = self.P_out.dims
        self.P_out_dims = self.P_union.broadcast_data(data, P_data=self.P_out)

        if len(self.P_in_dims) != len(self.P_out_dims):
            raise ValueError("Input and output partition must be same dimension.")

        # Share the two indices with every worker in the union.  The first
        # column of data contains the source index and the second contains
        # the destination index.
        local_indices = np.zeros(2, dtype=np.int)
        local_indices[0] = P_in.rank if P_in.active else -1
        local_indices[1] = P_out.rank if P_out.active else -1

        self.union_indices = self.P_union.allgather_data(local_indices)

    def _distdl_module_setup(self, input):

        # If we are not a worker, do nothing.
        if not self.P_union.active:
            return

        in_dims = self.P_in_dims
        out_dims = self.P_out_dims

        global_tensor_sizes = self._distdl_backend.compute_global_tensor_sizes(input[0],
                                                                               self.P_in,
                                                                               self.P_union)
        self.global_tensor_sizes = global_tensor_sizes

        tensor_dim = len(global_tensor_sizes)

        if len(in_dims) != tensor_dim:
            raise ValueError(f"Input partition mush have same dimension "
                             f"({len(in_dims)}) as input tensor rank ({tensor_dim}).")

        if len(out_dims) != tensor_dim:
            raise ValueError(f"Output partition mush have same dimension "
                             f"({len(out_dims)}) as input tensor rank ({tensor_dim}).")

        if 1 in global_tensor_sizes[global_tensor_sizes != out_dims]:
            raise ValueError(f"Input tensor must not be size 1 "
                             f"({global_tensor_sizes}) in a dimension where "
                             f"output partition is other than 1 ({out_dims}).")

        # We only need to move data to the output partition if we actually
        # have input data.  It is possible to have both input and output data,
        # either input or output data, or neither.  Hence the active guard.
        if self.P_in.active:
            in_coords = self.P_in.cartesian_coordinates(self.P_in.rank)

            # Compute our overlaps for each output subpartition.
            for rank, out_coords in enumerate(range_coords(out_dims)):
                sl = compute_partition_intersection(in_dims, in_coords,
                                                    out_dims, out_coords,
                                                    global_tensor_sizes)
                if sl is not None:
                    sz = compute_nd_slice_volume(sl)
                    # Reverse the mapping to get the output partner's rank in
                    # the common partition.
                    partner = np.where(self.union_indices[:, 1] == rank)[0][0]
                    self.in_data.append((sl, sz, partner))
                else:
                    self.in_data.append((None, None, None))

        # We only need to obtain data from the input partition if we actually
        # have output data.
        if self.P_out.active:
            out_coords = self.P_out.cartesian_coordinates(self.P_out.rank)

            # Compute our overlaps for each input subpartition.
            for rank, in_coords in enumerate(range_coords(in_dims)):
                sl = compute_partition_intersection(out_dims, out_coords,
                                                    in_dims, in_coords,
                                                    global_tensor_sizes)
                if sl is not None:
                    sz = compute_nd_slice_volume(sl)
                    # Reverse the mapping to get the input partner's rank in
                    # the common partition.
                    partner = np.where(self.union_indices[:, 0] == rank)[0][0]
                    self.out_data.append((sl, sz, partner))
                else:
                    self.out_data.append((None, None, None))

        buffs = self._allocate_buffers(self.dtype)
        self.in_buffers = buffs[0]
        self.out_buffers = buffs[1]

        self._distdl_is_setup = True
        self._input_shape = input[0].shape
        self._input_requires_grad = input[0].requires_grad

    def _distdl_module_teardown(self, input):

        # Reset all of the buffers and communication objects
        self.in_data = []
        self.out_data = []

        self.in_buffers = None
        self.out_buffers = None

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
