import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.unpadnd import UnpadNdFunction
from distdl.utilities.misc import DummyContext

P_world = MPIPartition(MPI.COMM_WORLD)

t = torch.zeros(5, 6)
print(f't =\n{t}')

ctx = DummyContext()
pad_width = [(1, 2), (1, 2)]

t_unpadded = UnpadNdFunction.forward(ctx, t, pad_width, value=0, partition=P_world)
print(f't_unpadded =\n{t_unpadded}')

t_padded = UnpadNdFunction.backward(ctx, t_unpadded)[0]
print(f't_padded =\n{t_padded}')
