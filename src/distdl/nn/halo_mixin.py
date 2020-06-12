import numpy as np

from distdl.utilities.slicing import compute_starts
from distdl.utilities.slicing import compute_subsizes


class HaloMixin:

    def _compute_exchange_info(self,
                               global_tensor_shape,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               partition_active,
                               partition_dims,
                               partition_coords):

        if not partition_active:
            return None, None, None, None

        dim = len(partition_dims)

        global_tensor_shape = np.atleast_1d(global_tensor_shape)
        kernel_size = np.atleast_1d(kernel_size)
        stride = np.atleast_1d(stride)
        padding = np.atleast_1d(padding)
        dilation = np.atleast_1d(dilation)

        def compute_lpad_length(array):
            return len(global_tensor_shape) - len(array)

        kernel_size = np.pad(kernel_size,
                             pad_width=(compute_lpad_length(kernel_size), 0),
                             mode='constant',
                             constant_values=1)
        stride = np.pad(stride,
                        pad_width=(compute_lpad_length(stride), 0),
                        mode='constant',
                        constant_values=1)
        padding = np.pad(padding,
                         pad_width=(compute_lpad_length(padding), 0),
                         mode='constant',
                         constant_values=0)
        dilation = np.pad(dilation,
                          pad_width=(compute_lpad_length(dilation), 0),
                          mode='constant',
                          constant_values=1)

        halo_sizes = self._compute_halo_sizes(partition_dims,
                                              partition_coords,
                                              global_tensor_shape,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation)

        recv_buffer_sizes = halo_sizes.copy()

        send_buffer_sizes = np.zeros_like(halo_sizes)

        for i in range(dim):
            lcoords = [x - 1 if j == i else x for j, x in enumerate(partition_coords)]
            nhalo = self._compute_halo_sizes(partition_dims,
                                             lcoords,
                                             global_tensor_shape,
                                             kernel_size,
                                             stride,
                                             padding,
                                             dilation)
            # If I have a left neighbor, my left send buffer size is my left
            # neighbor's right halo size
            if(lcoords[i] > -1):
                send_buffer_sizes[i, 0] = nhalo[i, 1]

            rcoords = [x + 1 if j == i else x for j, x in enumerate(partition_coords)]
            nhalo = self._compute_halo_sizes(partition_dims,
                                             rcoords,
                                             global_tensor_shape,
                                             kernel_size,
                                             stride,
                                             padding,
                                             dilation)
            # If I have a right neighbor, my right send buffer size is my right
            # neighbor's left halo size
            if(rcoords[i] < partition_dims[i]):
                send_buffer_sizes[i, 1] = nhalo[i, 0]

        local_input_tensor_shape = compute_subsizes(partition_dims, partition_coords, global_tensor_shape)
        halo_sizes_with_negatives = self._compute_halo_sizes(partition_dims,
                                                             partition_coords,
                                                             global_tensor_shape,
                                                             kernel_size,
                                                             stride,
                                                             padding,
                                                             dilation,
                                                             require_nonnegative=False)
        needed_ranges = self._compute_needed_ranges(local_input_tensor_shape, halo_sizes_with_negatives)

        halo_sizes = halo_sizes.astype(int)
        needed_ranges = needed_ranges.astype(int)

        return halo_sizes, recv_buffer_sizes, send_buffer_sizes, needed_ranges

    def _compute_needed_ranges(self, subsizes, halo_sizes):

        ranges = np.zeros_like(halo_sizes)

        # If we have a negative halo on the left, we want to not pass that
        # data to the torch layer
        ranges[:, 0] = -1*np.minimum(0, halo_sizes[:, 0])

        # The stop of the slice will be the data + the length of the two halos
        # and the last maximum is so that we dont shorten the stop (keeps the
        # parallel and sequential behavior exactly the same, but I dont think
        # it is strictly necessary)
        ranges[:, 1] = subsizes[:] + np.maximum(0, halo_sizes[:, 0]) + np.maximum(0, halo_sizes[:, 1])

        return ranges

    def _compute_out_sizes(self, in_sizes, kernel_size, stride, padding, dilation):
        return np.floor((in_sizes
                         + 2*padding
                         - dilation*(kernel_size-1) - 1)/stride + 1).astype(in_sizes.dtype)

    def _compute_halo_sizes(self,
                            dims,
                            coords,
                            global_input_tensor_shape,
                            kernel_size,
                            stride,
                            padding,
                            dilation,
                            require_nonnegative=True):

        global_input_tensor_shape = np.asarray(global_input_tensor_shape)

        local_input_tensor_shape = compute_subsizes(dims, coords, global_input_tensor_shape)
        local_input_tensor_starts = compute_starts(dims, coords, global_input_tensor_shape)

        # formula from pytorch docs for maxpool
        global_output_tensor_shape = self._compute_out_sizes(global_input_tensor_shape, kernel_size,
                                                             stride, padding, dilation)

        local_output_tensor_shape = compute_subsizes(dims, coords, global_output_tensor_shape)
        local_output_tensor_starts = compute_starts(dims, coords, global_output_tensor_shape)

        local_ouput_tensor_left_index = local_output_tensor_starts
        local_input_tensor_left_index_needed = self._compute_min_input_range(local_ouput_tensor_left_index,
                                                                             kernel_size,
                                                                             stride,
                                                                             padding,
                                                                             dilation)
        # Clamp to the boundary
        local_input_tensor_left_index_needed = np.maximum(np.zeros_like(global_input_tensor_shape), local_input_tensor_left_index_needed)

        local_output_tensor_right_index = local_output_tensor_starts + local_output_tensor_shape - 1
        local_input_tensor_right_index_needed = self._compute_max_input_range(local_output_tensor_right_index,
                                                                              kernel_size,
                                                                              stride,
                                                                              padding,
                                                                              dilation)
        # Clamp to the boundary
        local_input_tensor_right_index_needed = np.minimum(global_input_tensor_shape - 1, local_input_tensor_right_index_needed)

        # Compute the actual ghost values
        local_input_tensor_left_ghost = local_input_tensor_starts - local_input_tensor_left_index_needed
        local_input_tensor_right_ghost = local_input_tensor_right_index_needed - (local_input_tensor_starts + local_input_tensor_shape - 1)

        # Make sure the halos are always positive, so we get valid buffer sizes
        if require_nonnegative:
            local_input_tensor_left_ghost = np.maximum(local_input_tensor_left_ghost, 0)
            local_input_tensor_right_ghost = np.maximum(local_input_tensor_right_ghost, 0)

        return np.hstack([local_input_tensor_left_ghost, local_input_tensor_right_ghost]).reshape(2, -1).T
