__all__ = ["DistributedLinear"]

import torch

from distdl.nn.broadcast import Broadcast
from distdl.nn.module import Module
from distdl.nn.sum_reduce import SumReduce
from distdl.utilities.slicing import compute_subshape


class DistributedLinear(Module):
    r"""A distributed linear or affine layer.

    This class provides the user interface to a distributed linear layer.
    It utlizes back-end specific parallel data movement primitives but
    does not require its own back-end implementation.

    The base unit of work is given by the parition over the weight tensor.
    This class requires the following of the tensor partitions:

    1. :math:`P_x` over input tensor :math:`x` has shape :math:`1 \times
       P_{\text{f_in}}`.
    2. :math:`P_y` over input tensor :math:`y` has shape :math:`1 \times
       P_{\text{f_out}}`.
    3. :math:`P_W` over weight tensor :math:`W` has shape
       :math:`P_{\text{f_out}} \times P_{\text{f_in}}`.

    The bias term does not have its own partition.  The first dimension of the input and output partitions
    is the batch dimension and the second is the feature dimension.

    .. warning::
       This departs from PyTorch Linear layers, which allow intermediate
       dimensions in the tensors.

    Parameters
    ----------
    P_x :
        Partition of input tensor.
    P_y :
        Partition of output tensor.
    P_w :
        Partition of the weight tensor.
    in_features :
        Number of features in the *global* input tensor.
    out_features :
        Number of features in the *global* output tensor.
    bias : bool
        Indicates if a bias term should be used.

    """

    def __init__(self, P_x, P_y, P_w, in_features, out_features, bias=True):

        super(DistributedLinear, self).__init__()

        # P_x ~ 1 X P_fi
        self.P_x = P_x
        # P_y ~ 1 X P_fo
        self.P_y = P_y
        # P_w ~ P_fo X P_fi
        self.P_w = P_w

        # Bias flag
        self.bias = bias

        # Broadcast layer in the x-tensor
        self.x_broadcast = Broadcast(self.P_x, self.P_w, preserve_batch=True)

        # Each worker in P_W computes its own portion of the weight tensor and then
        # stores its own PyTorch Linear layer.  Only the 0th column of the tensor
        # also stores a bias.
        if self.P_w.active:
            local_in_features = compute_subshape(P_w.shape[1], P_w.index[1], in_features)
            local_out_features = compute_subshape(P_w.shape[0], P_w.index[0], out_features)
            # On column 0, use the specified bias, otherwise no bias to
            # prevent double counting
            bias = self.bias if (self.P_w.index[-1] == 0) else False
            self.sublinear = torch.nn.Linear(local_in_features[0], local_out_features[0], bias=bias)

        # Sum-reduce layer to get the y-tensor
        self.y_sum_reduce = SumReduce(self.P_w, self.P_y,
                                      transpose_src=True, preserve_batch=True)

    def forward(self, input):
        """Forward function interface.

        Parameters
        ----------
        input :
            Input tensor.

        """

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
