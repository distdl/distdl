import numpy as np
import torch


def zero_volume_tensor(b=None, dtype=None, requires_grad=False, device=None):

    if dtype is None:
        dtype = torch.get_default_dtype()

    if b is None:
        return torch.empty((0,), dtype=dtype, requires_grad=requires_grad, device=device)

    return torch.empty((b, 0), dtype=dtype, requires_grad=requires_grad, device=device)


class TensorStructure:
    """ Light-weight class to store and compare basic structure of Torch tensors.

    """

    def __init__(self, tensor=None):

        self.shape = None
        self.dtype = None
        self.requires_grad = None

        if tensor is not None:
            self.fill_from_tensor(tensor)

    def fill_from_tensor(self, tensor):

        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.requires_grad = tensor.requires_grad

    def __eq__(self, other):

        return ((self.shape == other.shape) and
                (self.dtype == other.dtype) and
                (self.requires_grad == other.requires_grad))


def distdl_padding_to_torch_padding(pad):
    r"""
    Accepts a NumPy ndarray describing the padding, and produces the torch F.pad format:
        [[a_0, b_0], ..., [a_n, b_n]]  ->  (a_n, b_n, ..., a_0, b_0)

    """
    return tuple(np.array(list(reversed(pad)), dtype=int).flatten())