import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.pooling import DistributedMaxPool1d
from distdl.utilities.debug import print_sequential
from distdl.utilities.slicing import compute_subsizes
from distdl.utilities.torch import NoneTensor

P_world = MPIPartition(MPI.COMM_WORLD)
P_world.comm.Barrier()

P = P_world.create_partition_inclusive(np.arange(3))
P_cart = P.create_cartesian_topology_partition([1, 1, 3])

global_tensor_sizes = np.array([1, 1, 10])

layer = DistributedMaxPool1d(global_tensor_sizes, P_cart, kernel_size=[2], stride=[2])

x = NoneTensor()
if P_cart.active:
    input_tensor_sizes = compute_subsizes(P_cart.dims, P_cart.cartesian_coordinates(P_cart.rank), global_tensor_sizes)
    x = torch.tensor(np.ones(shape=input_tensor_sizes) * (P_cart.rank + 1), dtype=float)
x.requires_grad = True

print_sequential(P_world.comm, f'rank = {P_world.rank}, input =\n{x}')

y = layer(x)

print_sequential(P_world.comm, f'rank = {P_world.rank}, output =\n{y}')
