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

    return slices


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


def range_index(shape):

    import itertools

    # An iterator that generates all cartesian coordinates over a set of
    # dimensions
    for x in itertools.product(*[range(y) for y in shape]):
        yield x
