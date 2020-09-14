import numpy as np

from distdl.utilities.dtype import torch_to_numpy_dtype_dict
from distdl.utilities.slicing import compute_nd_slice_volume


def allocate_halo_exchange_buffers(slices, recv_buffer_shape, send_buffer_shape, dtype):

    dim = len(slices)

    buffers = []

    numpy_dtype = torch_to_numpy_dtype_dict[dtype]

    for i in range(dim):
        lbb_len = compute_nd_slice_volume(slices[i][0]) if send_buffer_shape[i, 0] > 0 else 0
        lgb_len = compute_nd_slice_volume(slices[i][1]) if recv_buffer_shape[i, 0] > 0 else 0
        rbb_len = compute_nd_slice_volume(slices[i][2]) if send_buffer_shape[i, 1] > 0 else 0
        rgb_len = compute_nd_slice_volume(slices[i][3]) if recv_buffer_shape[i, 1] > 0 else 0

        buffers_i = [np.zeros(shape=x, dtype=numpy_dtype) if x > 0 else None for x in [lbb_len, lgb_len, rbb_len, rgb_len]]
        buffers.append(buffers_i)

    return buffers
