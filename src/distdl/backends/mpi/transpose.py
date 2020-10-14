import numpy as np

from distdl.utilities.dtype import torch_to_numpy_dtype_dict


def allocate_transpose_buffers(P_x_to_y_overlaps, P_y_to_x_overlaps, dtype):
    r"""Allocator for data movement buffers.

    Parameters
    ----------
    P_x_to_y_overlaps : list
        List of tuples (sz, sl, partner) for which current worker needs a send
        buffer.
    P_y_to_x_overlaps : list
        List of tuples (sz, sl, partner) for which current worker needs a
        receive buffer.
    dtype :
        Data type of input/output tensors.

    """

    numpy_dtype = torch_to_numpy_dtype_dict[dtype]

    # For each necessary copy, allocate send buffers.
    P_x_to_y_buffers = []
    for sl, sz, partner in P_x_to_y_overlaps:
        buff = None
        if sz is not None and partner != "self":
            buff = np.zeros(sz, dtype=numpy_dtype)

        P_x_to_y_buffers.append(buff)

    # For each necessary copy, allocate receive buffers.
    P_y_to_x_buffers = []
    for sl, sz, partner in P_y_to_x_overlaps:
        buff = None
        if sz is not None and partner != "self":
            buff = np.zeros(sz, dtype=numpy_dtype)

        P_y_to_x_buffers.append(buff)

    return P_x_to_y_buffers, P_y_to_x_buffers
