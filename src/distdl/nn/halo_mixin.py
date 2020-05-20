import numpy as np

from distdl.utilities.slicing import compute_starts
from distdl.utilities.slicing import compute_subsizes


class HaloMixin:

    def _compute_exchange_info(self,
                               x_in_sizes,
                               kernel_sizes,
                               strides,
                               pads,
                               dilations,
                               cartesian_parition):

        if not cartesian_parition.active:
            return None, None, None, None

        dim = cartesian_parition.dim
        dims = cartesian_parition.dims
        rank = cartesian_parition.rank
        coords = cartesian_parition.cartesian_coordinates(rank)

        x_in_sizes = np.asarray(x_in_sizes)
        kernel_sizes = np.asarray(kernel_sizes)
        strides = np.asarray(strides)
        pads = np.asarray(pads)
        dilations = np.asarray(dilations)

        halo_sizes = self._compute_halo_sizes(dims,
                                              coords,
                                              x_in_sizes,
                                              kernel_sizes,
                                              strides,
                                              pads,
                                              dilations)

        recv_buffer_sizes = halo_sizes.copy()

        send_buffer_sizes = np.zeros_like(halo_sizes)

        for i in range(dim):
            lcoords = [x - 1 if j == i else x for j, x in enumerate(coords)]
            nhalo = self._compute_halo_sizes(dims,
                                             lcoords,
                                             x_in_sizes,
                                             kernel_sizes,
                                             strides,
                                             pads,
                                             dilations)
            # If I have a left neighbor, my left send buffer size is my left
            # neighbor's right halo size
            if(lcoords[i] > -1):
                send_buffer_sizes[i, 0] = nhalo[i, 1]

            rcoords = [x + 1 if j == i else x for j, x in enumerate(coords)]
            nhalo = self._compute_halo_sizes(dims,
                                             rcoords,
                                             x_in_sizes,
                                             kernel_sizes,
                                             strides,
                                             pads,
                                             dilations)
            # If I have a right neighbor, my right send buffer size is my right
            # neighbor's left halo size
            if(rcoords[i] < dims[i]):
                send_buffer_sizes[i, 1] = nhalo[i, 0]

        x_in_subsizes = compute_subsizes(dims, coords, x_in_sizes)
        halo_sizes_with_negatives = self._compute_halo_sizes(dims,
                                                             coords,
                                                             x_in_sizes,
                                                             kernel_sizes,
                                                             strides,
                                                             pads,
                                                             dilations,
                                                             require_nonnegative=False)
        needed_ranges = self._compute_needed_ranges(x_in_subsizes, halo_sizes_with_negatives)

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

    def _compute_out_sizes(self, in_sizes, kernel_sizes, strides, pads, dilations):
        return np.floor((in_sizes
                         + 2*pads
                         - dilations*(kernel_sizes-1) - 1)/strides + 1).astype(in_sizes.dtype)

    def _compute_halo_sizes(self,
                            dims,
                            coords,
                            x_in_sizes,
                            kernel_sizes,
                            strides,
                            pads,
                            dilations,
                            require_nonnegative=True):

        x_in_sizes = np.asarray(x_in_sizes)

        x_in_subsizes = compute_subsizes(dims, coords, x_in_sizes)
        x_in_starts = compute_starts(dims, coords, x_in_sizes)

        # formula from pytorch docs for maxpool
        x_out_sizes = self._compute_out_sizes(x_in_sizes, kernel_sizes,
                                              strides, pads, dilations)

        x_out_subsizes = compute_subsizes(dims, coords, x_out_sizes)
        x_out_starts = compute_starts(dims, coords, x_out_sizes)

        local_left_indices = x_out_starts
        x_in_left_needed = self._compute_min_input_range(local_left_indices,
                                                         kernel_sizes,
                                                         strides,
                                                         pads,
                                                         dilations)
        # Clamp to the boundary
        x_in_left_needed = np.maximum(np.zeros_like(x_in_sizes), x_in_left_needed)

        local_right_indices = x_out_starts + x_out_subsizes - 1
        x_in_right_needed = self._compute_max_input_range(local_right_indices,
                                                          kernel_sizes,
                                                          strides,
                                                          pads,
                                                          dilations)
        # Clamp to the boundary
        x_in_right_needed = np.minimum(x_in_sizes - 1, x_in_right_needed)

        # Compute the actual ghost values
        x_in_left_ghost = x_in_starts - x_in_left_needed
        x_in_right_ghost = x_in_right_needed - (x_in_starts + x_in_subsizes - 1)

        # Make sure the halos are always positive, so we get valid buffer sizes
        if require_nonnegative:
            x_in_left_ghost = np.maximum(x_in_left_ghost, 0)
            x_in_right_ghost = np.maximum(x_in_right_ghost, 0)

        return np.hstack([x_in_left_ghost, x_in_right_ghost]).reshape(2, -1).T
