import numpy as np
import torch
from mpi4py import MPI


class BroadcastFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, comm, root):

        size = comm.Get_size()

        ctx.comm = comm
        ctx.size = size
        ctx.root = root

        if size == 1:
            return input.clone()

        input_numpy = input.detach().numpy()

        comm.Bcast(input_numpy, root=root)

        return torch.tensor(input_numpy, requires_grad=input.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):

        comm = ctx.comm
        size = ctx.size
        root = ctx.root

        if size == 1:
            return grad_output.clone(), None, None

        grad_output_numpy = grad_output.detach().numpy()
        reduced_data = np.zeros(shape=grad_output_numpy.shape, dtype=grad_output_numpy.dtype)

        comm.Reduce(grad_output_numpy, reduced_data, root=root, op=MPI.SUM)

        return torch.tensor(reduced_data, requires_grad=grad_output.requires_grad), None, None


class Broadcast(torch.nn.Module):

    def __init__(self, comm, root=0):
        super(Broadcast, self).__init__()

        self.comm = comm
        self.root = root

    def forward(self, input):
        return BroadcastFunction.apply(input, self.comm, self.root)
