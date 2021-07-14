import numpy as np
import torch
import torch.nn.functional as F


class UnpadNd(torch.nn.Module):

    def __init__(self, pad_width, value):

        super(UnpadNd, self).__init__()

        self.pad_width = pad_width
        self.value = value
        self.torch_pad = self._to_torch_padding(pad_width)
        self.slices = []
        for (lpad, rpad) in pad_width:
            start = lpad
            stop = -rpad if rpad > 0 else None
            self.slices.append(slice(start, stop, 1))

    def _to_torch_padding(self, pad):
        r"""
        Accepts a NumPy ndarray describing the padding, and produces the torch F.pad format:
            [[a_0, b_0], ..., [a_n, b_n]]  ->  (a_n, b_n, ..., a_0, b_0)
        """
        return tuple(np.array(list(reversed(pad)), dtype=int).flatten())

    def forward(self, input):
        return input[tuple(self.slices)]

    def backward(self, grad_output):
        return F.pad(grad_output, self.torch_pad, mode='constant', value=self.value)
