import numpy as np

INDEX_DTYPE = np.int64
MAX_INT = np.iinfo(INDEX_DTYPE).max
MIN_INT = np.iinfo(INDEX_DTYPE).min


def compute_subshape(P_shape, index, shape):

    P_shape = np.atleast_1d(P_shape)
    index = np.atleast_1d(index)
    shape = np.atleast_1d(shape)
    subshape = shape // P_shape
    subshape[index < shape % P_shape] += 1

    return subshape


def compute_start_index(P_shape, index, shape):

    P_shape = np.atleast_1d(P_shape)
    index = np.atleast_1d(index)
    shape = np.atleast_1d(shape)
    start_index = (shape // P_shape)*index
    start_index += np.minimum(index, shape % P_shape)

    return start_index


def compute_stop_index(P_shape, index, shape):

    start_index = compute_start_index(P_shape, index, shape)
    subshape = compute_subshape(P_shape, index, shape)
    stop_index = start_index + subshape

    return stop_index


def compute_intersection(r0_start_index, r0_stop_index,
                         r1_start_index, r1_stop_index):

    intersection_start_index = np.maximum(r0_start_index, r1_start_index)
    intersection_stop_index = np.minimum(r0_stop_index, r1_stop_index)
    intersection_subshape = intersection_stop_index - intersection_start_index
    intersection_subshape = np.maximum(intersection_subshape, 0)

    return intersection_start_index, intersection_stop_index, intersection_subshape


def assemble_slices(start_index, stop_index):

    slices = []

    for start, stop in zip(start_index, stop_index):
        slices.append(slice(start, stop, None))

    return tuple(slices)


def compute_partition_intersection(P_x_r_shape,
                                   P_x_r_index,
                                   P_x_s_shape,
                                   P_x_s_index,
                                   x_shape):

    # Extract the first subtensor description
    x_r_start_index = compute_start_index(P_x_r_shape, P_x_r_index, x_shape)
    x_r_stop_index = compute_stop_index(P_x_r_shape, P_x_r_index, x_shape)

    # Extract the second subtensor description
    x_s_start_index = compute_start_index(P_x_s_shape, P_x_s_index, x_shape)
    x_s_stop_index = compute_stop_index(P_x_s_shape, P_x_s_index, x_shape)

    # Compute the overlap between the subtensors and its volume
    x_i_start_index, x_i_stop_index, x_i_subshape = compute_intersection(x_r_start_index,
                                                                         x_r_stop_index,
                                                                         x_s_start_index,
                                                                         x_s_stop_index)
    x_i_volume = np.prod(x_i_subshape)

    # If the volume of the intersection is 0, we have no slice,
    # otherwise we need to determine the slices for x_i relative to
    # coordinates of x_r
    if x_i_volume == 0:
        return None
    else:
        x_i_start_index_rel_r = x_i_start_index - x_r_start_index
        x_i_stop_index_rel_r = x_i_start_index_rel_r + x_i_subshape
        x_i_slices_rel_r = assemble_slices(x_i_start_index_rel_r, x_i_stop_index_rel_r)
        return x_i_slices_rel_r


def compute_nd_slice_volume(slices):

    return np.prod([s.stop-s.start for s in slices])


def compute_nd_slice_shape(slices):
    r"""Computes the shape of a tuple of slices.

    This returns the shape of a given slice of a multi-dimensional array,
    tensor, or view.

    Parameters
    ----------
    slices : tuple of `slice` objects
        The volume to determine the shape of.

    Returns
    -------
    A tuple with the shape of the volume.

    """

    return tuple([s.stop-s.start for s in slices])


def range_index(shape):
    r"""An iterator over Cartesian indices of a given shape.

    Yields all possible Cartesian indices for a Partition of shape `shape` in
    row-major order: the first index varies slowest and the last index varies
    fastest.  For example, if the shape is (2, 3), the iterator will yield
    `[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]`.

    Parameters
    ----------
    shape : iterable
        Shape of partition.

    Yields
    ------
    A Cartesian index.

    """

    import itertools

    # An iterator that generates all cartesian coordinates over a set of
    # dimensions
    for x in itertools.product(*[range(y) for y in shape]):
        yield x


def worker_layout(shape):
    # Returns a numpy array with shape equal to the given shape such that
    # all values are set to the worker ranks.

    workers = np.zeros(shape.tolist(), dtype=int)
    for i, index in enumerate(range_index(shape)):
        workers[index] = i

    return workers


def filtered_range_index(shape, filter):
    r"""A filtered iterator over Cartesian indices of a given shape.

    Yields all possible Cartesian indices for a Partition of shape `shape`
    that match a given filter.  The filter is a set of integers that must
    match.  If a dimension is not required to match, the filter is `None` in
    that dimension.  For example, if the shape is (2, 3), and the filter is
    `(1, None)`, then the iterator will yield `[(1, 0), (1, 1), (1, 2)]`.

    Parameters
    ----------
    shape : iterable
        Shape of partition.
    filter : iterable
        Filter mask containing values to match (or None).

    Yields
    ------
    A Cartesian index.

    """

    def _filter(idx, filter):
        for i, f in zip(idx, filter):
            if f is None:
                continue
            if i != f:
                return False
        return True
    for idx_tuple in range_index(shape):
        if _filter(idx_tuple, filter):
            yield idx_tuple


def assemble_index_filter(index, dims, invert=False):
    r"""Helper for creating proper filters for `filtered_range_index()`.

    Given an iterable index and the dimensions that are required to match,
    creates the necessary filter.  This filter is the same as `index` except
    that it contains `None` for the indices not in `dims`.

    The `invert` flag produces the inverse mask from the same inputs.  That
    is, `None` for the indices in `dims` and the values of `index`
    elsewhere.

    For example, if `index` is `(4, 3, 2)` and `dims` is `(1, )`, the filter
    is `(None, 3, None)`.  If `invert` is true, then the filter would be
    `(4, None, 2)`.

    Parameters
    ----------
    index : iterable
        Index from which to build the filter.
    dims : iterable
        Dimensions of the index to preserve.
    invert : bool, optional
        Invert the `dims` when building the filter

    Returns
    -------
    Tuple containing filter constructed as above.

    """
    if invert:
        return tuple([index[k] if k not in dims else None for k in range(len(index))])
    else:
        return tuple([index[k] if k in dims else None for k in range(len(index))])
