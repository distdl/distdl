import numpy as np

from distdl.utilities.slicing import compute_start_index
from distdl.utilities.slicing import compute_subshape


class HaloMixin:

    def _compute_exchange_info(self,
                               x_global_shape,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               partition_active,
                               partition_shape,
                               partition_index,
                               subtensor_shapes=None):

        if not partition_active:
            return None, None, None, None

        dim = len(partition_shape)

        x_global_shape = np.atleast_1d(x_global_shape)
        kernel_size = np.atleast_1d(kernel_size)
        stride = np.atleast_1d(stride)
        padding = np.atleast_1d(padding)
        dilation = np.atleast_1d(dilation)

        def compute_lpad_length(array):
            return len(x_global_shape) - len(array)

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

        halo_shape = self._compute_halo_shape(partition_shape,
                                              partition_index,
                                              x_global_shape,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              subtensor_shapes=subtensor_shapes)

        recv_buffer_shape = halo_shape.copy()

        send_buffer_shape = np.zeros_like(halo_shape, dtype=int)

        for i in range(dim):
            lindex = [x - 1 if j == i else x for j, x in enumerate(partition_index)]
            # If I have a left neighbor, my left send buffer size is my left
            # neighbor's right halo size
            if lindex[i] > -1:
                nhalo = self._compute_halo_shape(partition_shape,
                                                 lindex,
                                                 x_global_shape,
                                                 kernel_size,
                                                 stride,
                                                 padding,
                                                 dilation,
                                                 subtensor_shapes=subtensor_shapes)
                send_buffer_shape[i, 0] = nhalo[i, 1]

            rindex = [x + 1 if j == i else x for j, x in enumerate(partition_index)]
            # If I have a right neighbor, my right send buffer size is my right
            # neighbor's left halo size
            if rindex[i] < partition_shape[i]:
                nhalo = self._compute_halo_shape(partition_shape,
                                                 rindex,
                                                 x_global_shape,
                                                 kernel_size,
                                                 stride,
                                                 padding,
                                                 dilation,
                                                 subtensor_shapes=subtensor_shapes)
                send_buffer_shape[i, 1] = nhalo[i, 0]

        if subtensor_shapes is not None:
            x_local_shape = subtensor_shapes[tuple(partition_index)]
        else:
            x_local_shape = compute_subshape(partition_shape, partition_index, x_global_shape)
        halo_shape_with_negatives = self._compute_halo_shape(partition_shape,
                                                             partition_index,
                                                             x_global_shape,
                                                             kernel_size,
                                                             stride,
                                                             padding,
                                                             dilation,
                                                             require_nonnegative=False,
                                                             subtensor_shapes=subtensor_shapes)
        needed_ranges = self._compute_needed_ranges(x_local_shape, halo_shape_with_negatives)

        halo_shape = halo_shape.astype(int)
        needed_ranges = needed_ranges.astype(int)

        return halo_shape, recv_buffer_shape, send_buffer_shape, needed_ranges

    def _compute_local_start_index(self,
                                   index,
                                   subtensor_shapes,
                                   x_local_shape):

        x_local_start_index = np.zeros_like(x_local_shape, dtype=np.int)
        dims = len(x_local_shape)
        for dim in range(dims):
            for i in range(index[dim]):
                idx = tuple(i if j == dim else 0 for j in range(dims))
                x_local_start_index[dim] += subtensor_shapes[idx][dim]
        return x_local_start_index

    def _compute_needed_ranges(self, tensor_shape, halo_shape):

        ranges = np.zeros_like(halo_shape, dtype=int)

        # If we have a negative halo on the left, we want to not pass that
        # data to the torch layer
        ranges[:, 0] = -1*np.minimum(0, halo_shape[:, 0])

        # The stop of the slice will be the data + the length of the two halos
        # and the last maximum is so that we dont shorten the stop (keeps the
        # parallel and sequential behavior exactly the same, but I dont think
        # it is strictly necessary)
        # TODO: Change this to correctly handle negative right halos.
        ranges[:, 1] = tensor_shape[:] + np.maximum(0, halo_shape[:, 0]) + np.maximum(0, halo_shape[:, 1])

        return ranges

    def _compute_out_shape(self, in_shape, kernel_size, stride, padding, dilation):
        return np.floor((in_shape
                         + 2*padding
                         - dilation*(kernel_size-1) - 1)/stride + 1).astype(in_shape.dtype)

    def _compute_halo_shape(self,
                            shape,
                            index,
                            x_global_shape,
                            kernel_size,
                            stride,
                            padding,
                            dilation,
                            require_nonnegative=True,
                            subtensor_shapes=None):

        x_global_shape = np.asarray(x_global_shape)

        # If subtensor_shapes is not None, then we cannot assume the input is balanced.
        if subtensor_shapes is not None:
            x_local_shape = subtensor_shapes[tuple(index)]
            x_local_start_index = self._compute_local_start_index(index,
                                                                  subtensor_shapes,
                                                                  x_local_shape)
        else:
            x_local_shape = compute_subshape(shape, index, x_global_shape)
            x_local_start_index = compute_start_index(shape, index, x_global_shape)

        # formula from pytorch docs for maxpool
        y_global_shape = self._compute_out_shape(x_global_shape, kernel_size,
                                                 stride, padding, dilation)

        y_local_shape = compute_subshape(shape, index, y_global_shape)
        y_local_start_index = compute_start_index(shape, index, y_global_shape)

        y_local_left_global_index = y_local_start_index
        x_local_left_global_index_needed = self._compute_min_input_range(y_local_left_global_index,
                                                                         kernel_size,
                                                                         stride,
                                                                         padding,
                                                                         dilation)
        # Clamp to the boundary
        x_local_left_global_index_needed = np.maximum(np.zeros_like(x_global_shape),
                                                      x_local_left_global_index_needed)

        y_local_right_global_index = y_local_start_index + y_local_shape - 1
        x_local_right_global_index_needed = self._compute_max_input_range(y_local_right_global_index,
                                                                          kernel_size,
                                                                          stride,
                                                                          padding,
                                                                          dilation)
        # Clamp to the boundary
        x_local_right_global_index_needed = np.minimum(x_global_shape - 1,
                                                       x_local_right_global_index_needed)

        # Compute the actual ghost values
        x_local_left_halo_shape = x_local_start_index - x_local_left_global_index_needed
        x_local_stop_index = x_local_start_index + x_local_shape - 1
        x_local_right_halo_shape = x_local_right_global_index_needed - x_local_stop_index

        # Make sure the halos are always positive, so we get valid buffer shape
        if require_nonnegative:
            x_local_left_halo_shape = np.maximum(x_local_left_halo_shape, 0)
            x_local_right_halo_shape = np.maximum(x_local_right_halo_shape, 0)

        return np.hstack([x_local_left_halo_shape, x_local_right_halo_shape]).reshape(2, -1).T
