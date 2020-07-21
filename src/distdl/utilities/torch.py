import torch


def zero_volume_tensor(b=None):

    if b is None:
        return torch.empty((0,))

    return torch.empty((b, 0))
