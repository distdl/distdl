import numpy as np


def compute_subtensor_shapes_unbalanced(local_tensor_structure, P_in, P_out=None):
    r"""Collectively assembles the shapes of all subtensors in a partition
        with no prior assumption on the distribution of elements.

        The output array has shape D+1, where D is the dimension of the tensor
        in question.  The first D dimensions of `shapes` correspond to the
        Cartesian decomposed workers and the last dimension corresponds to
        the shape of each subtensor.

        Parameters
        ----------
        local_tensor_structure : TensorStructure
            Structure of the local tensor, to be partially shared with P_out.
        P_in : MPIPartition
            Partition containing the local tensor.
        P_out : MPIPartition, optional
            Partition where an array of the local subtensor shapes will end up.

        Warning
        -------
        It is assumed that P_out has at least P_in's rank 0 in common.


        Returns
        -------
        Cartesian indexed NumPy array of all subtensors sizes on P_in.

    """

    shapes = None

    if P_in.active:

        shapes_size = list(P_in.shape) + [P_in.dim]
        shapes = np.zeros(shapes_size, dtype=np.int)

        # Generate a slice that isolates the worker's index into shapes
        # where it can store its shape
        sl = tuple([slice(i, i+1) for i in P_in.index] + [slice(None)])
        shapes[sl] = np.asarray(local_tensor_structure.shape)

        # Share everyone's shapes
        shapes = P_in.allreduce_data(shapes)

        # all max-reduce the shape

    if P_out is not None and P_out.active:
        shapes = P_out.broadcast_data(shapes, P_data=P_in)

    return shapes
