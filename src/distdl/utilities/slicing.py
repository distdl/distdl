import numpy as np

INDEX_DTYPE = np.int64
MAX_INT = np.iinfo(INDEX_DTYPE).max
MIN_INT = np.iinfo(INDEX_DTYPE).min


def compute_subshape(dims, coords, shape):

    dims = np.atleast_1d(dims)
    coords = np.atleast_1d(coords)
    shape = np.atleast_1d(shape)
    subshape = shape // dims
    subshape[coords < shape % dims] += 1

    return subshape


def compute_starts(dims, coords, shape):

    dims = np.atleast_1d(dims)
    coords = np.atleast_1d(coords)
    shape = np.atleast_1d(shape)
    starts = (shape // dims)*coords
    starts += np.minimum(coords, shape % dims)

    return starts


def compute_stops(dims, coords, shape):

    starts = compute_starts(dims, coords, shape)
    subshape = compute_subshape(dims, coords, shape)
    stops = starts + subshape

    return stops


def compute_intersection(r0_starts, r0_stops,
                         r1_starts, r1_stops):

    intersection_starts = np.maximum(r0_starts, r1_starts)
    intersection_stops = np.minimum(r0_stops, r1_stops)
    intersection_subshape = intersection_stops - intersection_starts
    intersection_subshape = np.maximum(intersection_subshape, 0)

    return intersection_starts, intersection_stops, intersection_subshape


def assemble_slices(starts, stops):

    slices = []

    for start, stop in zip(starts, stops):
        slices.append(slice(start, stop, None))

    return slices


def compute_partition_intersection(x_r_dims,
                                   x_r_coords,
                                   x_s_dims,
                                   x_s_coords,
                                   x_shape):

    # Extract the first subtensor description
    x_r_starts = compute_starts(x_r_dims, x_r_coords, x_shape)
    x_r_stops = compute_stops(x_r_dims, x_r_coords, x_shape)

    # Extract the second subtensor description
    x_s_starts = compute_starts(x_s_dims, x_s_coords, x_shape)
    x_s_stops = compute_stops(x_s_dims, x_s_coords, x_shape)

    # Compute the overlap between the subtensors and its volume
    x_i_starts, x_i_stops, x_i_subshape = compute_intersection(x_r_starts,
                                                               x_r_stops,
                                                               x_s_starts,
                                                               x_s_stops)
    x_i_volume = np.prod(x_i_subshape)

    # If the volume of the intersection is 0, we have no slice,
    # otherwise we need to determine the slices for x_i relative to
    # coordinates of x_r
    if x_i_volume == 0:
        return None
    else:
        x_i_starts_rel_r = x_i_starts - x_r_starts
        x_i_stops_rel_r = x_i_starts_rel_r + x_i_subshape
        x_i_slices_rel_r = assemble_slices(x_i_starts_rel_r, x_i_stops_rel_r)
        return x_i_slices_rel_r


def compute_nd_slice_volume(slices):

    return np.prod([s.stop-s.start for s in slices])


def range_coords(dims):

    import itertools

    # An iterator that generates all cartesian coordinates over a set of
    # dimensions
    for x in itertools.product(*[range(y) for y in dims]):
        yield x
