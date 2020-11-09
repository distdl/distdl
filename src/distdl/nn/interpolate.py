import torch

from distdl.functional.interpolate import InterpolateFunction


class Interpolate(torch.nn.Module):
    r"""A sequential interpolation layer.

    This class provides a user interface to a sequential interpolation layer,
    which, when the input is smaller than the output, mimics the PyTorch
    upsampling layer, up to one modification: input and output tensors are
    explicitely allowed to be subtensors of a larger tensor.

    Warning
    -------

    This layer should also allow downsampling interpolation, but this is
    currently untested behavior.

    Parameters
    ----------
    x_local_start : iterable
        Starting index (e.g., `start` in a Python slice) of the source subtensor.
    x_local_stop : iterable
        Stopping index (e.g., `stop` in a Python slice) of the source subtensor.
    x_global_shape : iterable
        Size of the global input tensor that the source subtensor is embedded in.
    y_local_start : iterable
        Starting index (e.g., `start` in a Python slice) of the destination subtensor.
    y_local_stop : iterable
        Stopping index (e.g., `stop` in a Python slice) of the destination subtensor.
    y_global_shape : iterable
        Size of the global input tensor that the destination subtensor is embedded in.
    scale_factor : optional
        Optional scale-factor representing a specific scaling used to obtain
        the relationship between `x_global_shape` and `y_global_shape`.  Used
        to match PyTorch UpSample behavior in the event that the specified
        scale factor does not produce an integer scaling between the source
        and destination tensors.
    mode : string, optional
        Interpolation mode.  Default is `'linear'`
    align_corners : bool, optional
        Analogous to PyTorch UpSample's `align_corner` flag.

    """

    def __init__(self,
                 x_local_start, x_local_stop, x_global_shape,
                 y_local_start, y_local_stop, y_global_shape,
                 scale_factor=None, mode='nearest', align_corners=False):

        super(Interpolate, self).__init__()

        self.mode = mode
        self.align_corners = align_corners

        self.scale_factor = scale_factor

        self.x_local_start = torch.Size(torch.as_tensor(x_local_start).squeeze())
        self.x_local_stop = torch.Size(torch.as_tensor(x_local_stop).squeeze())
        self.x_global_shape = torch.Size(torch.as_tensor(x_global_shape).squeeze())

        self.y_local_start = torch.Size(torch.as_tensor(y_local_start).squeeze())
        self.y_local_stop = torch.Size(torch.as_tensor(y_local_stop).squeeze())
        self.y_global_shape = torch.Size(torch.as_tensor(y_global_shape).squeeze())

    def forward(self, input):
        """Forward function interface.

        Parameters
        ----------
        input :
            Input tensor.

        """

        return InterpolateFunction.apply(input,
                                         self.scale_factor, self.mode, self.align_corners,
                                         self.x_local_start, self.x_local_stop, self.x_global_shape,
                                         self.y_local_start, self.y_local_stop, self.y_global_shape)
