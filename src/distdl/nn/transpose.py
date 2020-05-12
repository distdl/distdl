import torch

from distdl.utilities.slicing import compute_nd_slice_volume
from distdl.utilities.slicing import compute_partition_intersection


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

        in_slices = compute_partition_intersection(in_comm, out_comm, sizes)
        in_buffer_sizes = [None if s is None else
                           compute_nd_slice_volume(s) for s in in_slices]

        self.in_slices = in_slices
        self.in_buffer_sizes = in_buffer_sizes

        out_slices = compute_partition_intersection(out_comm, in_comm, sizes)
        out_buffer_sizes = [None if s is None else
                            compute_nd_slice_volume(s) for s in out_slices]

        self.out_slices = out_slices
        self.out_buffer_sizes = out_buffer_sizes

    def forward(self, input):

        # DistributedTransposeFunction.apply(input, self.in_buffers, self.out_buffers)
        pass
