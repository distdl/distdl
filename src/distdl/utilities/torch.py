import torch


class NoneTensor(torch.Tensor):

    def __init__(self):

        super(NoneTensor, self).__init__()

        self = torch.empty((0,))


def zero_volume_tensor(b=None):

    if b is None:
        return torch.empty((0,))

    return torch.empty((b, 0))
