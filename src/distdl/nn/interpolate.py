import torch

from distdl.functional.interpolate import PiecewiseConstantInterpolateFunction

interpolation_function = {
    'constant': PiecewiseConstantInterpolateFunction,
    'nearest': PiecewiseConstantInterpolateFunction,
}
#     'linear' : interpolate_ndtensor_linear_kernel,
#     'cubic' : interpolate_ndtensor_cubic_kernel
# }

class Interpolate(torch.nn.Module):

    def __init__(self, mode,
                 x_local_start, x_local_stop, x_global_shape,
                 y_local_start, y_local_stop, y_global_shape):

        super(Interpolate, self).__init__()

        self.mode = mode

        self.x_local_start = torch.Size(torch.as_tensor(x_local_start).squeeze())
        self.x_local_stop = torch.Size(torch.as_tensor(x_local_stop).squeeze())
        self.x_global_shape = torch.Size(torch.as_tensor(x_global_shape).squeeze())

        self.y_local_start = torch.Size(torch.as_tensor(y_local_start).squeeze())
        self.y_local_stop = torch.Size(torch.as_tensor(y_local_stop).squeeze())
        self.y_global_shape = torch.Size(torch.as_tensor(y_global_shape).squeeze())

        self.Function = interpolation_function[mode]

        self.kernel_args = (self.x_local_start, self.x_local_stop, self.x_global_shape,
                            self.y_local_start, self.y_local_stop, self.y_global_shape)

    def forward(self, input):

        return self.Function.apply(input, *self.kernel_args)
