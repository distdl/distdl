import numpy as np
import torch


class UnPadNdFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, pad_width, value):

        ctx.value = value
        ctx.pad_width = pad_width

        slices = []
        for (lpad, rpad) in pad_width:
            start = lpad
            stop = -rpad if rpad > 0 else None
            slices.append(slice(start, stop, 1))

        input_numpy = input.detach().numpy()

        result = input_numpy[tuple(slices)]

        return torch.tensor(result, requires_grad=input.requires_grad).float()

    @staticmethod
    def backward(ctx, grad_output):

        value = ctx.value
        pad_width = ctx.pad_width

        grad_output_numpy = grad_output.detach().numpy()

        result = np.pad(grad_output_numpy, pad_width, mode='constant', constant_values=value)

        return torch.tensor(result, requires_grad=grad_output.requires_grad).float(), None, None, None


class UnPadNd(torch.nn.Module):

    def __init__(self, pad_width, value):

        super(UnPadNd, self).__init__()

        self.pad_width = pad_width
        self.value = value

    def forward(self, input):
        return UnPadNdFunction.apply(input, self.pad_width, self.value)
