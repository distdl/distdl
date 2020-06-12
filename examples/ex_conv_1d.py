import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.conv import DistributedConv1d
from distdl.utilities.debug import print_sequential
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import NoneTensor

P_world = MPIPartition(MPI.COMM_WORLD)
P_world.comm.Barrier()

P = P_world.create_partition_inclusive(np.arange(3))
P_cart = P.create_cartesian_topology_partition([1, 1, 3])

x_global_shape = np.array([1, 1, 10])

layer = DistributedConv1d(P_cart, in_channels=1, out_channels=1, kernel_size=[3])

x = NoneTensor()
if P_cart.active:
    x_local_shape = compute_subshape(P_cart.dims,
                                     P_cart.cartesian_coordinates(P_cart.rank),
                                     x_global_shape)
    x = torch.Tensor(np.ones(shape=x_local_shape) * (P_cart.rank + 1))
x.requires_grad = True

print_sequential(P_world.comm, f'rank = {P_world.rank}, input =\n{x}')

y = layer(x)

print_sequential(P_world.comm, f'rank = {P_world.rank}, output =\n{y}')
