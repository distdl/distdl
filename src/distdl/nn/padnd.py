import numpy as np
import torch
import torch.nn.functional as F


class PadNd(torch.nn.Module):

    def __init__(self, pad_width, value):

        super(PadNd, self).__init__()
        self.pad_width = pad_width
        self.value = value
        self.torch_pad = self._to_torch_padding(self.pad_width)

    def _to_torch_padding(self, pad):
        r"""
        Accepts a NumPy ndarray describing the padding, and produces the torch F.pad format:
            [[a_0, b_0], ..., [a_n, b_n]]  ->  (a_n, b_n, ..., a_0, b_0)
        """
        return tuple(np.array(list(reversed(pad)), dtype=int).flatten())

    def forward(self, input):
        return F.pad(input, pad=self.torch_pad, mode='constant', value=self.value)
