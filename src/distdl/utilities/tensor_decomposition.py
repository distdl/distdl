import numpy as np

from distdl.utilities.slicing import assemble_slices
from distdl.utilities.slicing import compute_intersection
from distdl.utilities.slicing import range_index


def compute_subtensor_shapes_balanced(global_tensor_structure, P_tensor_shape):
    r"""Assembles the shapes of all subtensors in a partition, assuming a
        perfectly load balanced tensor decomposition.

        The output array has shape D+1, where D is the dimension of the tensor
        in question.  The first D dimensions of `shapes` correspond to the
        Cartesian decomposed workers and the last dimension corresponds to
        the shape of each subtensor.

        Parameters
        ----------
        global_tensor_structure : TensorStructure
            Structure of the global tensor.
        P_tensor_shape : MPIPartition
            Partition containing the global tensor.

        Returns
        -------
        Cartesian indexed NumPy array of all subtensors sizes of P_tensor.

    """

    P_tensor_shape = np.atleast_1d(P_tensor_shape)
    global_tensor_shape = np.atleast_1d(global_tensor_structure.shape)

    shapes_size = list(P_tensor_shape) + [len(P_tensor_shape)]
    shapes = np.zeros(shapes_size, dtype=np.int)

    for P_tensor_index in range_index(P_tensor_shape):

        P_tensor_index = np.atleast_1d(P_tensor_index)

        # Generate a slice that isolates the worker's index into shapes
        # where it can store its shape
        sl = tuple([slice(i, i+1) for i in P_tensor_index] + [slice(None)])

        subshape = global_tensor_shape // P_tensor_shape
        subshape[P_tensor_index < global_tensor_shape % P_tensor_shape] += 1

        shapes[sl] = np.asarray(subshape)

    return shapes


def compute_subtensor_start_indices(shapes):
    r"""Given Cartesian indexed array of subtensor shapes, computes the
        starting (left) index in the global tensor coordinate system.

        The input array has shape D+1, where D is the dimension of the tensor
        in question.  The first D dimensions of `shapes` correspond to the
        Cartesian decomposed workers and the last dimension corresponds to
        the shape of each subtensor.

        Parameters
        ----------
        shapes : np.ndarray
            D+1 dimensional array of subtensor shapes

        Returns
        -------
        Cartesian indexed NumPy array containing start (left) indices of all
        subtensors in the global tensor coordinate system.

    """

    indices = compute_subtensor_stop_indices(shapes)

    tensor_dims = len(shapes.shape)-1
    for d in range(tensor_dims):

        # Get the slice containing information for the first dimesion of indices
        sl_0 = [slice(None)]*tensor_dims + [slice(d, d+1)]
        sl_0[d] = slice(0, 1)
        sl_0 = tuple(sl_0)

        # Get the slice containing information for the first D-1 indices
        sl_a = [slice(None)]*tensor_dims + [slice(d, d+1)]
        sl_a[d] = slice(0, shapes.shape[d]-1)
        sl_a = tuple(sl_a)

        # Get the slice containing information for the last D-1 indices
        sl_b = [slice(None)]*tensor_dims + [slice(d, d+1)]
        sl_b[d] = slice(1, shapes.shape[d])
        sl_b = tuple(sl_b)

        # The starts of one subtensor are the stops of the previous one, so
        # these two slices effectively shift the values to the right
        indices[sl_b] = indices[sl_a]

        # The starts of the first subtensors are index 0
        indices[sl_0] = 0

    return indices


def compute_subtensor_stop_indices(shapes):
    r"""Given Cartesian indexed array of subtensor shapes, computes the
        stopping (right) index in the global tensor coordinate system.

        The input array has shape D+1, where D is the dimension of the tensor
        in question.  The first D dimensions of `shapes` correspond to the
        Cartesian decomposed workers and the last dimension corresponds to
        the shape of each subtensor.

        Parameters
        ----------
        shapes : np.ndarray
            D+1 dimensional array of subtensor shapes

        Returns
        -------
        Cartesian indexed NumPy array containing stop (right) indices of all
        subtensors in the global tensor coordinate system.

    """

    indices = shapes.copy()

    tensor_dims = len(shapes.shape)-1
    for d in range(tensor_dims):

        sl = tuple([slice(None)]*tensor_dims + [slice(d, d+1)])

        indices[sl] = np.cumsum(indices[sl], axis=d)

    return indices


def compute_subtensor_intersection_slice(x_start_index, x_stop_index,
                                         y_start_index, y_stop_index):
    r"""Given index bounds (start and stop indices), compute the overlap of
        two Cartesian indexed regions.

        The start and stop index sets of two D-dimensional regions, x and y,
        are sufficient to determine the indices of their overlaps, if any.

        Parameters
        ----------
        x_start_index : iterable
            D starting indices of the first tensor.
        x_stop_index : iterable
            D stopping indices of the first tensor.
        y_start_index : iterable
            D starting indices of the second tensor.
        y_stop_index : iterable
            D stopping indices of the second tensor.

        Returns
        -------
        Tuple of slice objects describing the overlap, relative to the first
        (x) tensor if there is non-zero overlap, `None` otherwise.

    """

    x_start_index = np.atleast_1d(x_start_index)
    x_stop_index = np.atleast_1d(x_stop_index)
    y_start_index = np.atleast_1d(y_start_index)
    y_stop_index = np.atleast_1d(y_stop_index)

    if (len(x_start_index.shape) != 1) or \
       (len(x_stop_index.shape) != 1) or \
       (len(y_start_index.shape) != 1) or \
       (len(y_stop_index.shape) != 1):
        raise ValueError("Index lists must be covnertable to 1-dimensional arrays.")

    # Compute the intersection between the x and y subtensors and its volume
    i_start_index, i_stop_index, i_shape = compute_intersection(x_start_index,
                                                                x_stop_index,
                                                                y_start_index,
                                                                y_stop_index)
    i_volume = np.prod(i_shape)

    # If the volume of the intersection is 0, we have no slice, otherwise we
    # need to determine the slices for the intersection relative to
    # coordinates of x.

    if i_volume == 0:
        return None
    else:
        i_start_index_rel_x = i_start_index - x_start_index
        i_stop_index_rel_x = i_start_index_rel_x + i_shape
        i_slice_rel_x = assemble_slices(i_start_index_rel_x, i_stop_index_rel_x)
        return i_slice_rel_x
