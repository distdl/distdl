import numpy as np
import torch

from distdl.utilities.torch import zero_volume_tensor


class ZeroVolumeCorrectorFunction(torch.autograd.Function):

    r"""Functional implementation of a wrapper for distributed loss outputs.

    Implements the required `forward()` and adjoint (`backward()`) operations
    in a consistent way that keeps PyTorch happy.

    Generally, PyTorch cannot backpropagate through zero-volume tensors, which
    are the default output when a distributed layer has no meaningful output.
    Thus, when an operation is at the end of a network, and a worker needs to
    return a zero-volume tensor, we actually have to return a valid tensor.  Yet,
    on the backward pass we must ignore any input and return the adjoint of the
    zero-volume output, which is a zero-volume grad input.

    For example, distributed loss functions assume that the loss value is
    reduced to rank 0.  Thus, rank 0 can return the actual loss but all other
    ranks would return nothing.  However, to satisfy PyTorch and Autograd's
    need for a scalar function in the root of the backward call, all other
    ranks must a tiny tensor from the forward call.  The backward
    call "reverses" this behavior, where rank 0 preserves the provided grad
    input but all other ranks inject the expected zero-volume tensor.

    DistDL distributed functions can handle zero-volume inputs to both forward
    and adjoint calls without this extra step.

    """

    @staticmethod
    def forward(ctx, input):

        r"""Forward function of zero-volume corrector wrapper.

        Parameters
        ----------
        ctx :
            PyTorch context.
        input : `torch.tensor`
            Input tensor.

        Returns
        -------
        output :
            `input` if it is not zero-volume, a tensor with a single element if the input is zero-volume.

        """

        ctx.sh = input.shape
        ctx.zero_volume = np.prod(input.shape) == 0

        if ctx.zero_volume:
            return torch.tensor(0.0, requires_grad=True, device=input.device).float()
        else:
            return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        r"""Adjoint function of zero-volume corrector wrapper.

        This method interfaces to the adjoint of the Jacobian of the
        forward  zero-volume corrector operation.

        Parameters
        ----------
        ctx :
            PyTorch context.
        grad_output : `torch.tensor`
            Input tensor.

        Returns
        -------
        output :
            `grad_output` if `input` was not zero-volume, a zero-volume tensor otherwise.

        """

        sh = ctx.sh

        if ctx.zero_volume:
            return zero_volume_tensor(sh[0], device=grad_output.device)
        else:
            return grad_output.clone()
