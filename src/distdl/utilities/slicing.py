import numpy as np

INDEX_DTYPE = np.int64
MAX_INT = np.iinfo(INDEX_DTYPE).max
MIN_INT = np.iinfo(INDEX_DTYPE).min

def compute_subsizes(dims, coords, sizes):

    sizes = np.asarray(sizes)
    subsizes = sizes // dims
    subsizes[coords < sizes % dims] += 1

    return subsizes


def compute_starts(dims, coords, sizes):

    starts = (sizes // dims)*coords
    starts += np.minimum(coords, sizes % dims)

    return starts


def compute_stops(dims, coords, sizes):

    starts = compute_starts(dims, coords, sizes)
    subsizes = compute_subsizes(dims, coords, sizes)
    stops = starts + subsizes

    return stops

def compute_intersection(r0_starts, r0_stops,
                         r1_starts, r1_stops):

    intersection_starts = np.maximum(r0_starts, r1_starts)
    intersection_stops = np.minimum(r0_stops, r1_stops)
    intersection_subsizes = intersection_stops - intersection_starts
    intersection_subsizes = np.maximum(intersection_subsizes, 0)

    return intersection_starts, intersection_stops, intersection_subsizes

def assemble_slices(starts, stops):

    slices = []

    for start, stop in zip(starts, stops):
        slices.append(slice(start, stop, None))

    return slices
