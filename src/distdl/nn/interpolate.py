import torch

from distdl.functional.interpolate import InterpolateFunction


class Interpolate(torch.nn.Module):

    def __init__(self,
                 x_local_start, x_local_stop, x_global_shape,
                 y_local_start, y_local_stop, y_global_shape,
                 mode='nearest', align_corners=False):

        super(Interpolate, self).__init__()

        self.mode = mode
        self.align_corners = align_corners

        self.x_local_start = torch.Size(torch.as_tensor(x_local_start).squeeze())
        self.x_local_stop = torch.Size(torch.as_tensor(x_local_stop).squeeze())
        self.x_global_shape = torch.Size(torch.as_tensor(x_global_shape).squeeze())

        self.y_local_start = torch.Size(torch.as_tensor(y_local_start).squeeze())
        self.y_local_stop = torch.Size(torch.as_tensor(y_local_stop).squeeze())
        self.y_global_shape = torch.Size(torch.as_tensor(y_global_shape).squeeze())

        self.kernel_args = (self.x_local_start, self.x_local_stop, self.x_global_shape,
                            self.y_local_start, self.y_local_stop, self.y_global_shape)

    def forward(self, input):

        return InterpolateFunction.apply(input, self.mode, self.align_corners, *self.kernel_args)
