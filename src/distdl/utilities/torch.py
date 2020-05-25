import torch


class NoneTensor(torch.Tensor):

    def __init__(self):

        super(NoneTensor, self).__init__()

        self = torch.empty((0,))
