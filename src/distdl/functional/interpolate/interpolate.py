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

    @staticmethod
    def forward(ctx, input, scale_factor, mode, align_corners, x_start, x_stop, x_global_shape, y_start, y_stop, y_global_shape):

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
