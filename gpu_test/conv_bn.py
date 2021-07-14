import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.conv_feature import DistributedFeatureConv2d
from distdl.nn.batchnorm import DistributedBatchNorm
from distdl.utilities.debug import print_sequential
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import zero_volume_tensor

torch.set_printoptions(linewidth=200)

device = torch.device('cuda:0')

P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

P = P_world.create_partition_inclusive(np.arange(4))
P_x = P.create_cartesian_topology_partition([1, 1, 2, 2])

x_global_shape = np.array([1, 1, 10, 10])

layer = DistributedBatchNorm(P_x, num_features=1)
layer = DistributedFeatureConv2d(P_x, in_channels=1, out_channels=1, kernel_size=[3, 3], padding=[1, 1])
layer = layer.to(device)

x = zero_volume_tensor(device=device)
if P_x.active:
    x_local_shape = compute_subshape(P_x.shape,
                                     P_x.index,
                                     x_global_shape)
    x = torch.ones(tuple(x_local_shape)) * (P_x.rank + 1)
    # x = x.to(device)
x.requires_grad = True

y_hat = torch.zeros_like(x)
y_hat = y_hat.to(device)

# print_sequential(P_world._comm, f'rank = {P_world.rank}, input =\n{x}')

y = layer(x)
y.backward(y_hat)

# print_sequential(P_world._comm, f'rank = {P_world.rank}, output =\n{x.grad}')
# print_sequential(P_world._comm, f'rank = {P_world.rank}, output =\n{y}')
