import torch

from distdl.functional.interpolate._cpp import constant_interpolation_adjoint
from distdl.functional.interpolate._cpp import constant_interpolation_forward


class PiecewiseConstantInterpolateFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, x_start, x_stop, x_global_shape, y_start, y_stop, y_global_shape):

        ctx.x_start = x_start
        ctx.x_stop = x_stop
        ctx.y_start = y_start
        ctx.x_global_shape = x_global_shape
        ctx.y_global_shape = y_global_shape

        y_shape = torch.as_tensor(y_stop) - torch.as_tensor(y_start)

        output = torch.zeros(*y_shape, dtype=input.dtype)

        constant_interpolation_forward(output, input,
                                       x_start, x_global_shape,
                                       y_start, y_global_shape
                                       )

        return output

    @staticmethod
    def backward(ctx, grad_output):

        x_start = ctx.x_start
        x_stop = ctx.x_stop
        y_start = ctx.y_start
        x_global_shape = ctx.x_global_shape
        y_global_shape = ctx.y_global_shape

        x_shape = torch.as_tensor(x_stop) - torch.as_tensor(x_start)

        grad_input = torch.zeros(*x_shape, dtype=grad_output.dtype)

        constant_interpolation_adjoint(grad_input, grad_output,
                                       x_start, x_global_shape,
                                       y_start, y_global_shape
                                       )

        return grad_input, None, None, None, None, None, None
