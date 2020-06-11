import numpy as np
import torch


class PadNdFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, pad_width, value):

        ctx.pad_width = pad_width

        input_numpy = input.detach().numpy()

        result = np.pad(input_numpy, pad_width, mode='constant', constant_values=value)

        return torch.tensor(result, requires_grad=input.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):

        pad_width = ctx.pad_width

        slices = []
        for (lpad, rpad) in pad_width:
            start = lpad
            stop = -rpad if rpad > 0 else None
            slices.append(slice(start, stop, 1))

        grad_output_numpy = grad_output.detach().numpy()

        result = grad_output_numpy[tuple(slices)]

        return torch.tensor(result, requires_grad=grad_output.requires_grad).float(), None, None, None


class PadNd(torch.nn.Module):

    def __init__(self, pad_width, value):

        super(PadNd, self).__init__()

        self.pad_width = pad_width
        self.value = value

    def forward(self, input):
        return PadNdFunction.apply(input, self.pad_width, self.value)
