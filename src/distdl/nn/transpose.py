import numpy as np
import torch

from distdl.utilities.slicing import assemble_slices
from distdl.utilities.slicing import compute_intersection
from distdl.utilities.slicing import compute_starts
from distdl.utilities.slicing import compute_stops


class DistributedTransposeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class DistributedTranspose(torch.nn.Module):

    def __init__(self, sizes, base_comm, in_comm, out_comm):
        super(DistributedTranspose, self).__init__()

        self.sizes = sizes
        self.base_comm = base_comm
        self.in_comm = in_comm
        self.out_comm = out_comm

        in_dims = in_comm.dims
        out_dims = out_comm.dims

        # Prep forward send / adjoint recv phase
        # Assumes input Cartesian communicator did not re-order ranks!
        # Should be same as base_rank
        in_rank = in_comm.Get_rank()
        in_coords = in_comm.Get_coords(in_rank)
        in_starts = compute_starts(in_dims, in_coords, sizes)
        in_stops = compute_stops(in_dims, in_coords, sizes)

        # Store slices of me
        in_slices = []
        # Store buffer sizes
        in_buffer_sizes = []

        # Build overlaps with output tensor
        for p_out in range(out_comm.Get_size()):
            p_out_coords = out_comm.Get_coords(p_out)
            p_out_starts = compute_starts(out_dims, p_out_coords, sizes)
            p_out_stops = compute_stops(out_dims, p_out_coords, sizes)

            i_starts, i_stops, i_subsizes = compute_intersection(in_starts, in_stops,
                                                                 p_out_starts, p_out_stops)

            p_buffer_size = np.prod(i_subsizes)
            if p_buffer_size == 0:
                in_slices.append(None)
                in_buffer_sizes.append(None)
            else:
                starts = i_starts - in_starts
                stops = starts + i_subsizes
                slices = assemble_slices(starts, stops)
                in_slices.append(slices)
                in_buffer_sizes.append(p_buffer_size)

        # Prep forward recv / adjoint send phase
        # Assumes input Cartesian communicator did not re-order ranks!
        # Should be same as base_rank
        out_rank = out_comm.Get_rank()
        out_coords = out_comm.Get_coords(out_rank)
        out_starts = compute_starts(out_dims, out_coords, sizes)
        out_stops = compute_stops(out_dims, out_coords, sizes)

        # Store slices of me
        out_slices = []
        # Store buffer sizes
        out_buffer_sizes = []

        # Build overlaps with output tensor
        for p_in in range(in_comm.Get_size()):
            p_in_coords = in_comm.Get_coords(p_in)
            p_in_starts = compute_starts(in_dims, p_in_coords, sizes)
            p_in_stops = compute_stops(in_dims, p_in_coords, sizes)

            i_starts, i_stops, i_subsizes = compute_intersection(out_starts, out_stops,
                                                                 p_in_starts, p_in_stops)

            p_buffer_size = np.prod(i_subsizes)
            if p_buffer_size == 0:
                out_slices.append(None)
                out_buffer_sizes.append(None)
            else:
                starts = i_starts - out_starts
                stops = starts + i_subsizes
                slices = assemble_slices(starts, stops)
                out_slices.append(slices)
                out_buffer_sizes.append(p_buffer_size)

    def forward(self, input):

        # DistributedTransposeFunction.apply(input, self.in_buffers, self.out_buffers)
        pass
