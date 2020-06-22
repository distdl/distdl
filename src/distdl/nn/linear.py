import torch

from distdl.nn.broadcast import Broadcast
from distdl.nn.module import Module
from distdl.nn.sum_reduce import SumReduce
from distdl.utilities.slicing import compute_subshape


class DistributedLinear(Module):

    def __init__(self, P_x, P_y, P_w, in_features, out_features, bias=True):

        super(DistributedLinear, self).__init__()

        # P_x ~ 1 X P_fi
        self.P_x = P_x
        # P_y ~ 1 X P_fo
        self.P_y = P_y
        # P_w ~ P_fo X P_fi
        self.P_w = P_w

        self.bias = bias

        self.x_broadcast = Broadcast(self.P_x, self.P_w, preserve_batch=True)

        if self.P_w.active:
            local_in_features = compute_subshape(P_w.shape[1], P_w.index[1], in_features)
            local_out_features = compute_subshape(P_w.shape[0], P_w.index[0], out_features)
            # On column 0, use the specified bias, otherwise no bias to
            # prevent double counting
            bias = self.bias if (self.P_w.index[-1] == 0) else False
            self.sublinear = torch.nn.Linear(local_in_features[0], local_out_features[0], bias=bias)

        self.y_sum_reduce = SumReduce(self.P_w, self.P_y,
                                      transpose_src=True, preserve_batch=True)

    def forward(self, input):

        if not (self.P_x.active or self.P_y.active or self.P_w.active):
            return input.clone()

        # broadcast x down the columns
        x = self.x_broadcast(input)

        # apply the linear layer
        if self.P_w.active:
            x = self.sublinear(x)

        # reduce y across the rows
        y = self.y_sum_reduce(x)

        return y
