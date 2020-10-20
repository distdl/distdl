import numpy as np

from distdl.utilities.dtype import torch_to_numpy_dtype_dict
from distdl.utilities.slicing import compute_nd_slice_shape


def allocate_halo_exchange_buffers(buffer_manager, slices, recv_buffer_shape, send_buffer_shape, dtype):

    dim = len(slices)

    buffers_out = []

    numpy_dtype = torch_to_numpy_dtype_dict[dtype]

    # Each dimension is performed sequentially.  Thus, we only need 4 buffers:
    # one each for left and right bulk and ghost.  The buffer shapes will be
    # viewed correctly for each dimension.
    count = 4

    buffers = buffer_manager.request_buffers(count, dtype=numpy_dtype)

    for i in range(dim):
        lbb_shape = compute_nd_slice_shape(slices[i][0]) if send_buffer_shape[i, 0] > 0 else 0
        lgb_shape = compute_nd_slice_shape(slices[i][1]) if recv_buffer_shape[i, 0] > 0 else 0
        rbb_shape = compute_nd_slice_shape(slices[i][2]) if send_buffer_shape[i, 1] > 0 else 0
        rgb_shape = compute_nd_slice_shape(slices[i][3]) if recv_buffer_shape[i, 1] > 0 else 0

        buffers_i = list()

        for j, shape in enumerate([lbb_shape, lgb_shape, rbb_shape, rgb_shape]):
            buff = None
            if np.prod(shape) > 0:
                buff = buffers[j]
                buff.allocate_view(shape)

            buffers_i.append(buff)

        buffers_out.append(buffers_i)

    return buffers_out
