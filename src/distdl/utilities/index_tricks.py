def cartesian_index_c(shape, index):
    idx = 0
    for d, c in zip(shape, index):
        idx = idx*d + c
    return idx


def cartesian_index_f(shape, index):
    idx = 0
    for d, c in zip(reversed(shape), reversed(index)):
        idx = idx*d + c
    return idx
