__all__ = ["Broadcast"]

from distdl.nn.module import Module
from distdl.utilities.torch import TensorStructure


class Broadcast(Module):
    r"""A distributed broadcast layer.

    This class provides the user interface to the broadcast distributed data
    movement primitive.  Implementation details are back-end specific.

    The Broadcast algorithm performs a broadcast from a tensor partitioned
    with by `P_x` to a new tensor partitioned with `P_y`.  The Broadcast
    requires the following:

    1. :math:`\text{dim}(P_x) \le \text{dim}(P_y)`.  If the
       :math:`\text{dim}(P_x) < \text{dim}(P_y)`, :math:`P_x` is implicitly
       extended *to the left* with ones.
    2. In a dimension where :math:`P_x` is not 1, :math:`P_y` must be equal to :math:`P_x`
       in that dimension.
    3. In a dimension where :math:`P_x` is 1, :math:`P_y` can take any positive value.

    The input tensor is broadcast (copied) along dimensions where :math:`P_x` is 1
    and and not broadcast along dimensions where :math:`P_x` matches :math:`P_y`.

    The `transpose_src` and `transpose_dest` optional arguments implicitly
    _reverse_ the shape of :math:`P_x` and :math:`P_y`, respectively, before the
    broadcast.  The input or output tensors are not transposed and the
    partition itself is not transposed on output.

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
    transpose_src : bool, optional
        Transpose the input partition prior to the broadcast.
    transpose_dest : bool, optional
        Transpose the output partition prior to the broadcast.
    preserve_batch : bool, optional
        Indicates if batch size should be preserved for zero-volume outputs.

    """

    def __init__(self, P_x, P_y,
                 transpose_src=False, transpose_dest=False,
                 preserve_batch=True):

        super(Broadcast, self).__init__()

        # Partition of input tensor.
        self.P_x = P_x

        # Partition of output tensor.
        self.P_y = P_y

        # Transpose the input partition prior to the broadcast.
        self.transpose_src = transpose_src

        # Transpose the output partition prior to the broadcast.
        self.transpose_dest = transpose_dest

        # Indicates if batch size should be preserved for zero-volume outputs.
        self.preserve_batch = preserve_batch

        # Indicates if broadcast requires any data movement.
        self.identity = False

        # Partition for sharing copy of local data.
        self.P_send = self._distdl_backend.Partition()

        # Partition for receiving copy of local data.
        self.P_recv = self._distdl_backend.Partition()

        # Other info needed by the functions

        # Structure of the input tensor (shape, dtype, requires_grad, etc).
        self.input_tensor_structure = TensorStructure()
        # Structure of the output tensor (shape, dtype, requires_grad, etc).
        self.output_tensor_structure = TensorStructure()

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

        # The identity case is if the partitions are of size 1,
        # or they are the same partition and neither is tranposed,
        # or they are the same partition and both are transposed.
        if self.P_x == self.P_y:
            if self.P_x.size == 1:
                self.identity = True
            elif (self.transpose_dest and self.transpose_src) or \
                 (not self.transpose_dest and not self.transpose_src):
                self.identity = True

    def _distdl_module_setup(self, input):
        r"""Broadcast module setup function.

        Constructs the necessary partition functions to implement the above
        described broadcast pattern.  This function performs collective
        communication across the input and output partitions.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        if not (self.P_x.active or self.P_y.active):
            return

        # If it is not an identity, we need actual Partitions to do the work.
        if not self.identity:
            bcast_partitions = self.P_x.create_broadcast_partition_to(self.P_y,
                                                                      self.transpose_src,
                                                                      self.transpose_dest)
            self.P_send = bcast_partitions[0]
            self.P_recv = bcast_partitions[1]

            self.input_tensor_structure = TensorStructure(input[0])
            self.output_tensor_structure = \
                self._distdl_backend.broadcast_tensor_structure(self.input_tensor_structure,
                                                                self.P_send,
                                                                self.P_recv)

        self._distdl_is_setup = True
        self._input_tensor_structure = TensorStructure(input[0])

    def _distdl_module_teardown(self, input):
        r"""Broadcast module teardown function.

        Nullifies the necessary partition functions.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        # Reset all of the buffers and communication objects
        self.P_send.deactivate()
        self.P_recv.deactivate()

        # Reset any data stored about the tensor
        self.input_tensor_structure = TensorStructure()
        self.output_tensor_structure = TensorStructure()

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

        Function = self._distdl_backend.functional.broadcast.BroadcastFunction

        # If this is an identity operation (no communication necessary),
        # simply return a clone of the input.
        if self.identity:
            return input.clone()

        # If this worker is not active for the input or output, then the input
        # should be a zero-volume tensor, and the output should be the same.
        if not (self.P_x.active or self.P_y.active):
            return input.clone()

        return Function.apply(input,
                              self.P_send,
                              self.P_recv,
                              self.preserve_batch,
                              self.input_tensor_structure,
                              self.output_tensor_structure)
