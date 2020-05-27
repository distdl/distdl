def cartesian_index_c(dims, coords):
    idx = 0
    for d, c in zip(dims, coords):
        idx = idx*d + c
    return idx


def cartesian_index_f(dims, coords):
    idx = 0
    for d, c in zip(reversed(dims), reversed(coords)):
        idx = idx*d + c
    return idx
