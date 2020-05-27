import torch

from distdl.nn.broadcast import Broadcast
from distdl.nn.sum_reduce import SumReduce


class DistributedLinear(torch.nn.Module):

    def __init__(self, P_x, x_sizes, P_y, y_sizes, P_mul, bias=True):

        super(DistributedLinear, self).__init__()

        self.P_x = P_x
        self.x_sizes = x_sizes
        self.P_y = P_y
        self.y_sizes = y_sizes

        self.P_mul = P_mul

        self.bias = bias

        self.x_broadcast = Broadcast(self.P_x, self.P_mul)
        if self.P_mul.active:
            # On column 0, use the specified bias, otherwise no bias to
            # prevent double counting
            bias = self.bias if (self.P_mul.coords[-1] == 0) else False
            self.sublinear = torch.nn.Linear(x_sizes[-1], y_sizes[-1], bias=bias)
        self.y_sum_reduce = SumReduce(self.P_mul, self.P_y, transpose_src=True)

    def forward(self, input):

        # broadcast x down the columns
        x = self.x_broadcast(input)

        # apply the linear layer
        if self.P_mul.active:
            x = self.sublinear(x)

        # reduce y across the rows
        y = self.y_sum_reduce(x)

        return y
