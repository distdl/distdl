import numpy as np

from distdl.nn.module import Module
from distdl.utilities.slicing import compute_nd_slice_volume
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.slicing import range_index
from distdl.utilities.tensor_decomposition import compute_subtensor_intersection_slice
from distdl.utilities.tensor_decomposition import compute_subtensor_shapes_balanced
from distdl.utilities.tensor_decomposition import compute_subtensor_start_indices
from distdl.utilities.tensor_decomposition import compute_subtensor_stop_indices
from distdl.utilities.torch import TensorStructure


class DistributedTranspose(Module):
    r"""A distributed transpose layer.

    This class provides the user interface to the transpose distributed data
    movement primitive.  Implementation details are back-end specific.

    The Transpose algorithm performs a transpose, shuffle, or generalized
    all-to-all from a tensor partitioned with by `P_x` to a new tensor
    partitioned with `P_y`.  The values of the tensor do not change.  Only the
    distribution of the tensor over the workers changes.

    If ``P_x`` and ``P_y`` are exactly equal, then no data movement occurs.

    For input and output tensors that have a batch dimension, the batch
    dimension needs to be preserved.  If a tensor does not have a batch
    dimension, we should not preserve that for zero-volume outputs.  The
    `preserve_batch` option controls this.

    Parameters
    ----------
    P_x :
        Partition of input tensor.
    P_y :
        Partition of output tensor.
    preserve_batch : bool, optional
        Indicates if batch size should be preserved for zero-volume outputs.

    """

    def __init__(self, P_x, P_y, preserve_batch=True):
        super(DistributedTranspose, self).__init__()

        # Global structure of the input tensor, assembled when layer is called
        self.global_input_tensor_structure = TensorStructure()

        # Local input and output tensor structures, defined when layer is called
        self.input_tensor_structure = TensorStructure()
        self.output_tensor_structure = TensorStructure()

        # Partition of input tensor.
        self.P_x = P_x

        # Partition of output tensor.
        self.P_y = P_y

        # Indicates if batch size should be preserved for zero-volume outputs.
        self.preserve_batch = preserve_batch

        # List of meta data describing copies of subvolumes of input tensor
        # out of the current worker
        self.P_x_to_y_overlaps = []

        # List of meta data describing copies of subvolumes of output tensor
        # into the current worker
        self.P_y_to_x_overlaps = []

        # List of buffers for copying data to other workers
        self.P_x_to_y_buffers = None

        # List of buffers for copying data from other workers
        self.P_y_to_x_buffers = None

        # Indicates if transpose requires any data movement.
        self.identity = False

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

        # If the two partitions are the same, no further information is
        # required.
        if P_x == P_y:
            self.identity = True
            return

        # Otherwise, we need the union of the input and output partitions
        # so that data can be copied across them.
        P_union = self._distdl_backend.Partition()
        if P_x.active or P_y.active:
            P_union = P_x.create_partition_union(P_y)
        self.P_union = P_union

        # Setup these variables incase the current worker is inactive in
        # the union.
        self.P_x_shape = None
        self.P_y_shape = None
        self.union_indices = None

        if not P_union.active:
            return

        # All active workers need the shapes of both partitions so that buffer
        # sizes and subtensor overlaps can be computed.
        data = None
        if self.P_x.active:
            data = self.P_x.shape
        self.P_x_shape = self.P_union.broadcast_data(data, P_data=self.P_x)

        data = None
        if self.P_y.active:
            data = self.P_y.shape
        self.P_y_shape = self.P_union.broadcast_data(data, P_data=self.P_y)

        if len(self.P_x_shape) != len(self.P_y_shape):
            raise ValueError("Input and output partition must be same dimension.")

        # Share the two indices with every worker in the union.  The first
        # column of data contains the source index and the second contains
        # the destination index.
        data = np.array([P_x.rank if P_x.active else -1], dtype=np.int)
        self.P_x_ranks = P_union.allgather_data(data)

        data = np.array([P_y.rank if P_y.active else -1], dtype=np.int)
        self.P_y_ranks = P_union.allgather_data(data)

        # Get some types and functions from the back-end
        self.allocate_transpose_buffers = self._distdl_backend.transpose.allocate_transpose_buffers

    def _distdl_module_setup(self, input):
        r"""Transpose module setup function.

        Constructs the necessary buffers and meta information about outbound
        and inbound copies to each worker.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        self._distdl_is_setup = True
        self._input_tensor_structure.fill_from_tensor(input[0])

        # If we are not an active worker, do nothing.
        if not self.P_union.active:
            return

        self.input_tensor_structure = TensorStructure(input[0])

        self.global_input_tensor_structure = \
            self._distdl_backend.assemble_global_tensor_structure(self.input_tensor_structure,
                                                                  self.P_x,
                                                                  self.P_union)
        x_global_shape = self.global_input_tensor_structure.shape

        if self.P_y.active:
            self.output_tensor_structure.shape = compute_subshape(self.P_y.shape,
                                                                  self.P_y.index,
                                                                  x_global_shape)

        tensor_dim = len(x_global_shape)

        if len(self.P_x_shape) != tensor_dim:
            raise ValueError(f"Input partition mush have same dimension "
                             f"({len(self.P_x_shape)}) as input tensor rank ({tensor_dim}).")

        if len(self.P_y_shape) != tensor_dim:
            raise ValueError(f"Output partition mush have same dimension "
                             f"({len(self.P_y_shape)}) as input tensor rank ({tensor_dim}).")

        if 1 in x_global_shape[x_global_shape != self.P_y_shape]:
            raise ValueError(f"Input tensor must not be size 1 "
                             f"({x_global_shape}) in a dimension where "
                             f"output partition is other than 1 ({self.P_y_shape}).")

        # Get the collective input lengths and origins. This may be load
        # balanced or it may not be.  Therefore we will always assume is is
        # not load balanced and just build the subshape tensor manually.
        # This output is needed everywhere so it goes to P_union.
        compute_subtensor_shapes_unbalanced = \
            self._distdl_backend.tensor_decomposition.compute_subtensor_shapes_unbalanced
        x_subtensor_shapes = compute_subtensor_shapes_unbalanced(self.input_tensor_structure,
                                                                 self.P_x,
                                                                 self.P_union)

        # Get the collective output lengths and origins. This will always be
        # load balanced, so we can infer the subshape tensor from the global
        # tensor shape and the shape of P_y.  At this point, every worker in
        # P_union has both of these pieces of information, so we can build it
        # with no communication.

        y_subtensor_shapes = compute_subtensor_shapes_balanced(self.global_input_tensor_structure,
                                                               self.P_y_shape)

        # Given all subtensor shapes, we can compute the start and stop indices
        # for each partition.

        x_subtensor_start_indices = compute_subtensor_start_indices(x_subtensor_shapes)
        x_subtensor_stop_indices = compute_subtensor_stop_indices(x_subtensor_shapes)

        y_subtensor_start_indices = compute_subtensor_start_indices(y_subtensor_shapes)
        y_subtensor_stop_indices = compute_subtensor_stop_indices(y_subtensor_shapes)

        # We only need to move data to the output partition if we actually
        # have input data.  It is possible to have both input and output data,
        # either input or output data, or neither.  Hence the active guard.
        if self.P_x.active:
            x_slice = tuple([slice(i, i+1) for i in self.P_x.index] + [slice(None)])
            x_start_index = x_subtensor_start_indices[x_slice].squeeze()
            x_stop_index = x_subtensor_stop_indices[x_slice].squeeze()

            # Compute our overlaps for each output subpartition.
            for rank, P_y_index in enumerate(range_index(self.P_y_shape)):

                y_slice = tuple([slice(i, i+1) for i in P_y_index] + [slice(None)])
                y_start_index = y_subtensor_start_indices[y_slice].squeeze()
                y_stop_index = y_subtensor_stop_indices[y_slice].squeeze()

                sl = compute_subtensor_intersection_slice(x_start_index, x_stop_index,
                                                          y_start_index, y_stop_index)

                if sl is not None:
                    sz = compute_nd_slice_volume(sl)
                    # Reverse the mapping to get the output partner's rank in
                    # the common partition.
                    partner = np.where(self.P_y_ranks == rank)[0][0]
                    self.P_x_to_y_overlaps.append((sl, sz, partner))
                else:
                    self.P_x_to_y_overlaps.append((None, None, None))

        # We only need to obtain data from the input partition if we actually
        # have output data.
        if self.P_y.active:
            y_slice = tuple([slice(i, i+1) for i in self.P_y.index] + [slice(None)])
            y_start_index = y_subtensor_start_indices[y_slice].squeeze()
            y_stop_index = y_subtensor_stop_indices[y_slice].squeeze()

            # Compute our overlaps for each input subpartition.
            for rank, P_x_index in enumerate(range_index(self.P_x_shape)):

                x_slice = tuple([slice(i, i+1) for i in P_x_index] + [slice(None)])
                x_start_index = x_subtensor_start_indices[x_slice].squeeze()
                x_stop_index = x_subtensor_stop_indices[x_slice].squeeze()

                sl = compute_subtensor_intersection_slice(y_start_index, y_stop_index,
                                                          x_start_index, x_stop_index)

                if sl is not None:
                    sz = compute_nd_slice_volume(sl)
                    # Reverse the mapping to get the input partner's rank in
                    # the common partition.
                    partner = np.where(self.P_x_ranks == rank)[0][0]
                    self.P_y_to_x_overlaps.append((sl, sz, partner))
                else:
                    self.P_y_to_x_overlaps.append((None, None, None))

        buffs = self.allocate_transpose_buffers(self.P_x_to_y_overlaps,
                                                self.P_y_to_x_overlaps,
                                                self.global_input_tensor_structure.dtype)
        self.P_x_to_y_buffers = buffs[0]
        self.P_y_to_x_buffers = buffs[1]

    def _distdl_module_teardown(self, input):
        r"""Transpose module teardown function.

        Deallocates buffers safely.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        # Reset all of the buffers and communication objects
        self.P_x_to_y_overlaps = []
        self.P_y_to_x_overlaps = []

        self.P_x_to_y_buffers = None
        self.P_y_to_x_buffers = None

        # Reset any info about the input
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

    def _distdl_input_changed(self, input):
        r"""Determine if the structure of inputs has changed.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        new_tensor_structure = TensorStructure(input[0])

        return self._input_tensor_structure != new_tensor_structure

    def forward(self, input):
        """Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to be broadcast.

        """

        Function = self._distdl_backend.autograd.transpose.DistributedTransposeFunction

        # If this is an identity operation (no communication necessary),
        # simply return a clone of the input.
        if self.identity:
            return input.clone()

        # If this worker is not active for the input or output, then the input
        # should be a zero-volume tensor, and the output should be the same.
        if not (self.P_x.active or self.P_y.active):
            return input.clone()

        return Function.apply(input,
                              self.P_union,
                              self.global_input_tensor_structure,
                              self.input_tensor_structure,
                              self.output_tensor_structure,
                              self.P_x,
                              self.P_x_to_y_overlaps,
                              self.P_x_to_y_buffers,
                              self.P_y,
                              self.P_y_to_x_overlaps,
                              self.P_y_to_x_buffers,
                              self.preserve_batch)
