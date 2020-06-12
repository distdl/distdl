import numpy as np

from distdl.nn.module import Module
from distdl.utilities.slicing import compute_nd_slice_volume
from distdl.utilities.slicing import compute_partition_intersection
from distdl.utilities.slicing import range_coords


class DistributedTranspose(Module):

    def __init__(self, P_x, P_y):
        super(DistributedTranspose, self).__init__()

        self.global_tensor_shape = None

        self.P_x = P_x
        self.P_y = P_y

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

        if P_x == P_y:
            self.identity = True
            return

        P_union = self._distdl_backend.Partition()
        if P_x.active or P_y.active:
            P_union = P_x.create_partition_union(P_y)
        self.P_union = P_union

        self.P_x_dims = None
        self.P_y_dims = None
        self.union_indices = None

        if not P_union.active:
            return

        data = None
        if self.P_x.active:
            data = self.P_x.dims
        self.P_x_dims = self.P_union.broadcast_data(data, P_data=self.P_x)

        data = None
        if self.P_y.active:
            data = self.P_y.dims
        self.P_y_dims = self.P_union.broadcast_data(data, P_data=self.P_y)

        if len(self.P_x_dims) != len(self.P_y_dims):
            raise ValueError("Input and output partition must be same dimension.")

        # Share the two indices with every worker in the union.  The first
        # column of data contains the source index and the second contains
        # the destination index.
        local_indices = np.zeros(2, dtype=np.int)
        local_indices[0] = P_x.rank if P_x.active else -1
        local_indices[1] = P_y.rank if P_y.active else -1

        self.union_indices = self.P_union.allgather_data(local_indices)

    def _distdl_module_setup(self, input):

        # If we are not a worker, do nothing.
        if not self.P_union.active:
            return

        in_dims = self.P_x_dims
        out_dims = self.P_y_dims

        global_tensor_shape = self._distdl_backend.compute_global_tensor_shape(input[0],
                                                                               self.P_x,
                                                                               self.P_union)
        self.global_tensor_shape = global_tensor_shape

        tensor_dim = len(global_tensor_shape)

        if len(in_dims) != tensor_dim:
            raise ValueError(f"Input partition mush have same dimension "
                             f"({len(in_dims)}) as input tensor rank ({tensor_dim}).")

        if len(out_dims) != tensor_dim:
            raise ValueError(f"Output partition mush have same dimension "
                             f"({len(out_dims)}) as input tensor rank ({tensor_dim}).")

        if 1 in global_tensor_shape[global_tensor_shape != out_dims]:
            raise ValueError(f"Input tensor must not be size 1 "
                             f"({global_tensor_shape}) in a dimension where "
                             f"output partition is other than 1 ({out_dims}).")

        # We only need to move data to the output partition if we actually
        # have input data.  It is possible to have both input and output data,
        # either input or output data, or neither.  Hence the active guard.
        if self.P_x.active:
            in_coords = self.P_x.cartesian_coordinates(self.P_x.rank)

            # Compute our overlaps for each output subpartition.
            for rank, out_coords in enumerate(range_coords(out_dims)):
                sl = compute_partition_intersection(in_dims, in_coords,
                                                    out_dims, out_coords,
                                                    global_tensor_shape)
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
        if self.P_y.active:
            out_coords = self.P_y.cartesian_coordinates(self.P_y.rank)

            # Compute our overlaps for each input subpartition.
            for rank, in_coords in enumerate(range_coords(in_dims)):
                sl = compute_partition_intersection(out_dims, out_coords,
                                                    in_dims, in_coords,
                                                    global_tensor_shape)
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
                              self.global_tensor_shape,
                              self.P_x,
                              self.in_data,
                              self.in_buffers,
                              self.P_y,
                              self.out_data,
                              self.out_buffers,
                              self.dtype)
