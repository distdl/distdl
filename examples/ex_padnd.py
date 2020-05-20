import torch

import distdl.nn.padnd as padnd
from distdl.utilities.misc import DummyContext

t = torch.ones(3, 4)
print(f't =\n{t}')

ctx = DummyContext()
pad_width = [(1, 2), (3, 4)]

t_padded = padnd.PadNdFunction.forward(ctx, t, pad_width, value=0)
print(f't_padded =\n{t_padded}')

t_unpadded = padnd.PadNdFunction.backward(ctx, t_padded)[0]
print(f't_unpadded =\n{t_unpadded}')
