import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.pooling import DistributedAvgPool1d
from distdl.utilities.debug import print_sequential
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import NoneTensor

P_world = MPIPartition(MPI.COMM_WORLD)
P_world.comm.Barrier()

P_x_base = P_world.create_partition_inclusive(np.arange(3))
P_x = P_x_base.create_cartesian_topology_partition([1, 1, 3])

x_global_shape = np.array([1, 1, 10])

layer = DistributedAvgPool1d(P_x, kernel_size=[2], stride=[2])

x = NoneTensor()
if P_x.active:
    x_local_shape = compute_subshape(P_x.dims,
                                     P_x.coords,
                                     x_global_shape)
    x = torch.tensor(np.ones(shape=x_local_shape) * (P_x.rank + 1), dtype=float)
x.requires_grad = True

print_sequential(P_world.comm, f'rank = {P_world.rank}, input =\n{x}')

y = layer(x)

print_sequential(P_world.comm, f'rank = {P_world.rank}, output =\n{y}')
