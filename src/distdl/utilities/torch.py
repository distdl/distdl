import torch


def zero_volume_tensor(b=None):

    if b is None:
        return torch.empty((0,))

    return torch.empty((b, 0))


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
