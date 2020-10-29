import torch

interpolation_kernels = {
    'constant' = interpolate_ndtensor_constant_kernel,
    'linear' = interpolate_ndtensor_linear_kernel,
    'cubic' = interpolate_ndtensor_cubic_kernel
}


class PiecewiseConstantInterpolateFunction(torch.autograd.Function):


    @staticmethod
    def forward(ctx, input, x_starts, x_stops, y_starts, y_stops):

        ctx.x_starts = x_starts
        ctx.x_stops = x_stops
        ctx.y_starts = y_starts
        ctx.y_stops = y_stops

        y_shape = y_stops - y_starts

        output = torch.zeros(tuple(y_shape), input.dtype)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        x_starts = ctx.x_starts
        x_stops = ctx.x_stops
        y_starts = ctx.y_starts
        y_stops = ctx.y_stops

        x_shape = x_stops - x_starts

        grad_input = torch.zeros(tuple(x_shape), grad_output.dtype)

        return grad_input





class Interpolate(torch.nn.Module):

    def __init__(self, mode, x_starts, x_stops, y_starts, y_stops):

        super(Interpolate, self).__init__()

        self.mode = mode

        self.x_starts = x_starts.squeeze()
        self.x_stops = x_stops.squeeze()

        self.y_starts = y_starts.squeeze()
        self.y_stops = y_stops.squeeze()

        self.y_shape = self.y_stops - self.y_starts

        self.kernel = interpolation_kernels[mode]

        self.kernel_args = (self.x_starts, self.x_stops, self.y_starts, self.y_stops)

    def forward(self, input):


        return self.kernel(input, output, *self.kernel_args)
