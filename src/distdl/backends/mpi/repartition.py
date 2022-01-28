from distdl.utilities.dtype import torch_to_numpy_dtype_dict


def allocate_repartition_buffers(buffer_manager, P_x_to_y_overlaps, P_y_to_x_overlaps, dtype):
    r"""Allocator for data movement buffers.

    Parameters
    ----------
    P_x_to_y_overlaps : list
        List of tuples (sl, sh, partner) for which current worker needs a send
        buffer.
    P_y_to_x_overlaps : list
        List of tuples (sl, sh, partner) for which current worker needs a
        receive buffer.
    dtype :
        Data type of input/output tensors.

    """

    numpy_dtype = torch_to_numpy_dtype_dict[dtype]

    # count the buffers we need
    count = 0
    for sl, sh, partner in P_x_to_y_overlaps:
        if sl is not None and partner != "self":
            count += 1
    for sl, sh, partner in P_y_to_x_overlaps:
        if sl is not None and partner != "self":
            count += 1

    buffers = buffer_manager.request_buffers(count, dtype=numpy_dtype)

    i = 0

    # For each necessary copy, allocate send buffers.
    P_x_to_y_buffers = []
    for sl, sh, partner in P_x_to_y_overlaps:
        buff = None
        if sl is not None and partner != "self":
            buff = buffers[i]
            buff.allocate_view(sh)
            i += 1

        P_x_to_y_buffers.append(buff)

    # For each necessary copy, allocate receive buffers.
    P_y_to_x_buffers = []
    for sl, sh, partner in P_y_to_x_overlaps:
        buff = None
        if sl is not None and partner != "self":
            buff = buffers[i]
            buff.allocate_view(sh)
            i += 1

        P_y_to_x_buffers.append(buff)

    return P_x_to_y_buffers, P_y_to_x_buffers
