def cartesian_index_c(shape, coords):
    idx = 0
    for d, c in zip(shape, coords):
        idx = idx*d + c
    return idx


def cartesian_index_f(shape, coords):
    idx = 0
    for d, c in zip(reversed(shape), reversed(coords)):
        idx = idx*d + c
    return idx
