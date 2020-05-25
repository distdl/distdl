def cartesian_index(dims, coords):
    idx = 0
    for d, c in zip(dims, coords):
        idx = idx*d + c
    return idx
