import numpy as np
import torch
from mpi4py import MPI


class BroadcastFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, partition, root):

        ctx.partition = partition
        ctx.root = root

        if partition.size == 1:
            return input.clone()

        if partition.active:
            input_numpy = input.detach().numpy()

            partition.comm.Bcast(input_numpy, root=root)

            return torch.tensor(input_numpy,
                                requires_grad=input.requires_grad)
        else:
            return None

    @staticmethod
    def backward(ctx, grad_output):

        partition = ctx.partition
        root = ctx.root

        if partition.size == 1:
            return grad_output.clone(), None, None

        if partition.active:
            grad_output_numpy = grad_output.detach().numpy()
            reduced_data = np.zeros(shape=grad_output_numpy.shape,
                                    dtype=grad_output_numpy.dtype)

            partition.comm.Reduce(grad_output_numpy, reduced_data,
                                  root=root, op=MPI.SUM)

            return torch.tensor(reduced_data,
                                requires_grad=grad_output.requires_grad), None, None
        else:
            return None, None, None


class Broadcast(torch.nn.Module):

    def __init__(self, partition, root=0):
        super(Broadcast, self).__init__()

        self.partition = partition
        self.root = root

    def forward(self, input):
        return BroadcastFunction.apply(input, self.partition, self.root)
