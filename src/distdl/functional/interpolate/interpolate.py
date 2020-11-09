import torch

from distdl.functional.interpolate._cpp import constant_interpolation_adjoint
from distdl.functional.interpolate._cpp import constant_interpolation_forward
from distdl.functional.interpolate._cpp import linear_interpolation_adjoint
from distdl.functional.interpolate._cpp import linear_interpolation_forward

fwd_functions = {
    'nearest': constant_interpolation_forward,
    'linear': linear_interpolation_forward,
}
adj_functions = {
    'nearest': constant_interpolation_adjoint,
    'linear': linear_interpolation_adjoint,
}


class InterpolateFunction(torch.autograd.Function):
    r"""Functional implementation of a general interpolation layer.

    Implements the required `forward()` and adjoint (`backward()`) operations
    for piecewise constant (nearest-left neighbor) and piecewise linear
    interpolation.

    Warning
    -------
    This implementation currently requires that tensors have data stored in main
    memory (CPU) only, not auxiliary memories such as those on GPUs.

    """

    @staticmethod
    def forward(ctx, input,
                scale_factor, mode, align_corners,
                x_start, x_stop, x_global_shape,
                y_start, y_stop, y_global_shape):
        r"""Forward function of interpolation  layer.

        Currently, only `"nearest"` and `"linear"` are valid interpolation
        modes.

        Parameters
        ----------
        ctx :
            PyTorch context.
        input : `torch.tensor`
            Input tensor.
        x_local_start : torch.Tensor
            Starting index (e.g., `start` in a Python slice) of the source subtensor.
        x_local_stop : torch.Tensor
            Stopping index (e.g., `stop` in a Python slice) of the source subtensor.
        x_global_shape : torch.Tensor
            Size of the global input tensor that the source subtensor is embedded in.
        y_local_start : torch.Tensor
            Starting index (e.g., `start` in a Python slice) of the destination subtensor.
        y_local_stop : torch.Tensor
            Stopping index (e.g., `stop` in a Python slice) of the destination subtensor.
        y_global_shape : torch.Tensor
            Size of the global input tensor that the destination subtensor is embedded in.
        scale_factor :
            Scale-factor representing a specific scaling used to obtain
            the relationship between `x_global_shape` and `y_global_shape`.  Used
            to match PyTorch UpSample behavior in the event that the specified
            scale factor does not produce an integer scaling between the source
            and destination tensors.
        mode : string
            Interpolation mode.
        align_corners : bool
            Analogous to PyTorch UpSample's `align_corner` flag.

        Returns
        -------
        output :
            Output tensor.

        """

        if scale_factor is None:
            scale_factor = -1

        ctx.scale_factor = scale_factor

        ctx.mode = mode
        ctx.align_corners = align_corners

        ctx.x_start = x_start
        ctx.x_stop = x_stop
        ctx.y_start = y_start
        ctx.x_global_shape = x_global_shape
        ctx.y_global_shape = y_global_shape

        y_shape = torch.as_tensor(y_stop) - torch.as_tensor(y_start)
        output = torch.zeros(*y_shape, dtype=input.dtype)

        fwd_functions[mode](output, input,
                            x_start, x_global_shape,
                            y_start, y_global_shape,
                            scale_factor,
                            align_corners)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        r"""Adjoint function of interpolation layer.

        This method interfaces to the adjoint of the Jacobian of the used
        forward interpolation algorithm.

        Parameters
        ----------
        ctx :
            PyTorch context.
        grad_output : `torch.tensor`
            Input tensor.

        Returns
        -------
        output :
            Output tensor.

        """

        scale_factor = ctx.scale_factor

        mode = ctx.mode
        align_corners = ctx.align_corners

        x_start = ctx.x_start
        x_stop = ctx.x_stop
        y_start = ctx.y_start
        x_global_shape = ctx.x_global_shape
        y_global_shape = ctx.y_global_shape

        x_shape = torch.as_tensor(x_stop) - torch.as_tensor(x_start)

        grad_input = torch.zeros(*x_shape, dtype=grad_output.dtype)

        adj_functions[mode](grad_input, grad_output,
                            x_start, x_global_shape,
                            y_start, y_global_shape,
                            scale_factor,
                            align_corners)

        return grad_input, None, None, None, None, None, None, None, None, None
