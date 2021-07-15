from distdl.nn.module import Module
from distdl.utilities.torch import TensorStructure


class AllSumReduce(Module):
    r"""A distributed all-sum-reduce layer.

    This class provides the user interface to the all-sum-reduction
    distributed data movement primitive.  Implementation details are back-end
    specific.

    The AllSumReduce algorithm performs an all-sum-reduction within a single
    partition. Thus, the standard DistDL sum-reduction/broadcast rules are
    implicitly satisfied.

    Functionally, the input tensor is reduced along dimensions specified by
    the `axes_reduce` field and the result of that reduction is broadcast
    along the same dimensions.  However, the underlying implementation will
    not typically apply these two operations directly.

    One of `axes_reduce` or `axes_keep`, only, may be set.

    Parameters
    ----------
    P_x :
        Partition of input and output tensor.
    axes_reduce : tuple, optional
        Partition dimensions along which the all-reduction takes place.
    axes_keep : tuple, optional
        Partition dimensions to reduce to.  Complement of `axes_reduce`.

    """

    def __init__(self, P_x, axes_reduce=None, axes_keep=None):

        super(AllSumReduce, self).__init__()

        # Partition of input and output tensor.
        self.P_x = P_x

        # Partition dimensions along which the all-reduction takes place.
        # While we compute both terms, `axes_reduce` is used internally.
        if axes_reduce is None and axes_keep is None:
            raise ValueError("One of `axes_reduce` or `axes_keep` must be specified.")
        elif axes_reduce is not None and axes_keep is not None:
            raise ValueError("Only one of `axes_reduce` or `axes_keep` may be specified.")
        elif axes_reduce is not None:
            self.axes_reduce = axes_reduce
            self.axes_keep = [d for d in range(P_x.dim) if d not in axes_reduce]
        elif axes_keep is not None:
            self.axes_reduce = [d for d in range(P_x.dim) if d not in axes_keep]
            self.axes_keep = axes_keep

        # Indicates if broadcast requires any data movement.
        self.identity = False

        # Partition for performing all-reduction.
        self.P_allreduce = self._distdl_backend.Partition()

        # Structure of the input tensor (shape, dtype, requires_grad, etc).
        self.input_tensor_structure = TensorStructure()
        # Structure of the output tensor (shape, dtype, requires_grad, etc).
        self.output_tensor_structure = TensorStructure()

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

        # The identity case is if the partition is of size 1,
        if self.P_x.size == 1:
            self.identity = True

    def _distdl_module_setup(self, input):
        r"""AllSumReduce module setup function.

        Constructs the necessary partition functions to implement the above
        described reduction pattern.  This function performs collective
        communication across the input and output partitions.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        if not (self.P_x.active):
            return

        # If it is not an identity, we need actual Partitions to do the work.
        if not self.identity:

            self.P_allreduce = self.P_x.create_allreduction_partition(self.axes_reduce)

            self.input_tensor_structure = TensorStructure(input[0])
            self.output_tensor_structure = self.input_tensor_structure

        self._distdl_is_setup = True
        self._input_tensor_structure = TensorStructure(input[0])

    def _distdl_module_teardown(self, input):
        r"""AllSumReduce module teardown function.

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
        self.P_allreduce.deactivate()

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
            Input tensor to be all-sum-reduced.

        """

        Function = self._distdl_backend.functional.all_sum_reduce.AllSumReduceFunction

        if self.identity:
            return input.clone()

        if not (self.P_x.active):
            return input.clone()

        return Function.apply(input,
                              self.P_allreduce,
                              self.input_tensor_structure,
                              self.output_tensor_structure)
