import numpy as np
import torch

from distdl.utilities.tensor_decomposition import compute_subtensor_shapes_balanced
from distdl.utilities.tensor_decomposition import compute_subtensor_start_indices
from distdl.utilities.tensor_decomposition import compute_subtensor_stop_indices


class InterpolateMixin:
    r"""A mixin providing general support for distributed layers using
    Interpolation.

    """

    def _compute_needed_start(self, y_global_idx,
                              x_global_shape, y_global_shape,
                              scale_factor, mode, align_corners):

        # Finds the start index required to get to y_global_idx

        y_global_idx = torch.as_tensor(y_global_idx)
        y_global_shape = torch.as_tensor(y_global_shape).to(torch.float64)
        x_global_shape = torch.as_tensor(x_global_shape).to(torch.float64)

        if mode == "nearest":

            fac = torch.ones_like(x_global_shape)
            fac[2:] = x_global_shape[2:] / y_global_shape[2:]

            idx = torch.floor(fac*(y_global_idx))
            idx = idx.to(torch.int64)

        elif mode == "linear":

            fac = torch.ones_like(x_global_shape)
            if align_corners:
                fac[2:] = (x_global_shape[2:] - 1) / (y_global_shape[2:] - 1)
                idx = torch.floor(fac*(y_global_idx))
            else:
                # This calculation should match exactly interpolaate.h and the
                # ATen interpolation code.
                if scale_factor is not None:
                    fac[2:] = 1.0 / scale_factor
                else:
                    fac[2:] = (x_global_shape[2:]) / (y_global_shape[2:])

                idx = fac*(y_global_idx + 0.5) - 0.5
                idx = torch.where(idx < 0, torch.zeros_like(idx), idx)
                idx = torch.floor(idx)
            idx = idx.to(torch.int64)
        else:
            raise NotImplementedError(f"Mode `{mode}` is not supported.")

        idx = torch.where(idx < 0, torch.zeros_like(idx), idx)

        return idx

    def _compute_needed_stop(self, y_global_idx,
                             x_global_shape, y_global_shape,
                             scale_factor, mode, align_corners):

        # Finds the stop index required to get to y_global_idx

        y_global_idx = torch.as_tensor(y_global_idx)
        y_global_shape = torch.as_tensor(y_global_shape).to(torch.float64)
        x_global_shape = torch.as_tensor(x_global_shape).to(torch.float64)

        if mode == "nearest" or mode == "constant":

            fac = torch.ones_like(x_global_shape)
            fac[2:] = x_global_shape[2:] / y_global_shape[2:]

            idx = torch.floor(fac*(y_global_idx))
            idx = idx.to(torch.int64)

        elif mode == "linear":

            fac = torch.ones_like(x_global_shape)
            if align_corners:
                fac[2:] = (x_global_shape[2:] - 1) / (y_global_shape[2:] - 1)
                idx = torch.floor(fac*(y_global_idx))
            else:
                # This calculation should match exactly interpolaate.h and the
                # ATen interpolation code.
                if scale_factor is not None:
                    fac[2:] = 1.0 / scale_factor
                else:
                    fac[2:] = (x_global_shape[2:]) / (y_global_shape[2:])

                idx = fac*(y_global_idx + 0.5) - 0.5
                idx = torch.where(idx < 0, torch.zeros_like(idx), idx)
                idx = torch.floor(idx)
            idx = (idx+1).to(torch.int64)
        else:
            raise NotImplementedError(f"Mode `{mode}` is not supported.")

        # idx is the actual index, so we need to add 1 to get to a python slice stop
        idx += 1

        idx = torch.where(idx > x_global_shape, x_global_shape.to(torch.int64), idx)

        return idx

    def _compute_halo_shape(self,
                            x_local_start, x_local_stop, x_global_shape,
                            y_local_start, y_local_stop, y_global_shape,
                            scale_factor, mode, align_corners,
                            require_nonnegative=True):

        x_local_start_needed = self._compute_needed_start(y_local_start,
                                                          x_global_shape,
                                                          y_global_shape,
                                                          scale_factor,
                                                          mode,
                                                          align_corners)
        x_local_start_needed = np.maximum(np.zeros_like(x_global_shape),
                                          x_local_start_needed)

        x_local_stop_needed = self._compute_needed_stop(y_local_stop - 1,
                                                        x_global_shape,
                                                        y_global_shape,
                                                        scale_factor,
                                                        mode,
                                                        align_corners)
        x_local_stop_needed = np.minimum(x_global_shape,
                                         x_local_stop_needed)

        # Compute the actual ghost values
        x_local_left_halo_shape = x_local_start - x_local_start_needed
        x_local_right_halo_shape = x_local_stop_needed - x_local_stop

        # Make sure the halos are always positive, so we get valid buffer shape
        if require_nonnegative:
            x_local_left_halo_shape = np.maximum(x_local_left_halo_shape, 0)
            x_local_right_halo_shape = np.maximum(x_local_right_halo_shape, 0)

        halo_sizes = np.hstack([x_local_left_halo_shape, x_local_right_halo_shape]).reshape(2, -1).T

        return halo_sizes

    def _compute_needed_ranges(self, tensor_shape, halo_shape):

        ranges = np.zeros_like(halo_shape)

        # If we have a negative halo on the left, we want to not pass that
        # data to the torch layer
        ranges[:, 0] = -1*np.minimum(0, halo_shape[:, 0])

        # The stop of the slice will be the data + the length of the two halos
        # This allows negative right halos
        ranges[:, 1] = tensor_shape[:] + np.maximum(0, halo_shape[:, 0]) + halo_shape[:, 1]

        return ranges

    def _compute_exchange_info(self, P_x,
                               x_global_tensor_structure, y_global_tensor_structure,
                               scale_factor, mode, align_corners):

        if not P_x.active:
            return None, None, None, None

        dim = len(P_x.shape)

        x_slice = tuple([slice(i, i+1) for i in P_x.index] + [slice(None)])

        x_subtensor_shapes = compute_subtensor_shapes_balanced(x_global_tensor_structure,
                                                               P_x.shape)
        x_subtensor_start_indices = compute_subtensor_start_indices(x_subtensor_shapes)
        x_subtensor_stop_indices = compute_subtensor_stop_indices(x_subtensor_shapes)

        y_subtensor_shapes = compute_subtensor_shapes_balanced(y_global_tensor_structure,
                                                               P_x.shape)
        y_subtensor_start_indices = compute_subtensor_start_indices(y_subtensor_shapes)
        y_subtensor_stop_indices = compute_subtensor_stop_indices(y_subtensor_shapes)

        def halo_shape_args(index, require_nonnegative=True):
            _slice = tuple([slice(i, i+1) for i in index] + [slice(None)])
            _x_start_index = torch.from_numpy(x_subtensor_start_indices[_slice].squeeze())
            _x_stop_index = torch.from_numpy(x_subtensor_stop_indices[_slice].squeeze())
            _y_start_index = torch.from_numpy(y_subtensor_start_indices[_slice].squeeze())
            _y_stop_index = torch.from_numpy(y_subtensor_stop_indices[_slice].squeeze())

            args = (_x_start_index, _x_stop_index, x_global_tensor_structure.shape,
                    _y_start_index, _y_stop_index, y_global_tensor_structure.shape,
                    scale_factor, mode, align_corners, require_nonnegative)

            return args

        halo_shape = self._compute_halo_shape(*halo_shape_args(P_x.index))

        recv_buffer_shape = halo_shape.copy()
        send_buffer_shape = np.zeros_like(halo_shape)

        for i in range(dim):
            lindex = [x - 1 if j == i else x for j, x in enumerate(P_x.index)]
            nhalo = self._compute_halo_shape(*halo_shape_args(lindex))

            # If I have a left neighbor, my left send buffer size is my left
            # neighbor's right halo size
            if(lindex[i] > -1):
                send_buffer_shape[i, 0] = nhalo[i, 1]

            rindex = [x + 1 if j == i else x for j, x in enumerate(P_x.index)]
            nhalo = self._compute_halo_shape(*halo_shape_args(rindex))

            # If I have a right neighbor, my right send buffer size is my right
            # neighbor's left halo size
            if(rindex[i] < P_x.shape[i]):
                send_buffer_shape[i, 1] = nhalo[i, 0]

        x_local_shape = torch.from_numpy(x_subtensor_shapes[x_slice].squeeze())

        # Allow the negatives to get the slice range
        halo_shape_with_negatives = self._compute_halo_shape(*halo_shape_args(P_x.index, False))

        needed_ranges = self._compute_needed_ranges(x_local_shape, halo_shape_with_negatives)

        halo_shape = halo_shape.astype(int)
        needed_ranges = needed_ranges.astype(int)

        return halo_shape, recv_buffer_shape, send_buffer_shape, needed_ranges
