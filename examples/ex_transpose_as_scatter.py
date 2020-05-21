import numpy as np
import torch
from mpi4py import MPI

import distdl.nn.transpose as transpose
from distdl.backends.mpi.partition import MPIPartition
from distdl.utilities.debug import print_sequential

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
ranks = np.arange(P_world.size)

in_dims = (1, 1)
out_dims = (2, 2)

in_size = np.prod(in_dims)
out_size = np.prod(out_dims)

# In partition is the first in_size ranks
in_ranks = ranks[:in_size]
P_in = P_world.create_subpartition(in_ranks)
P_in_cart = P_in.create_cartesian_subpartition(in_dims)

# Out partition is the last outsize ranks
out_ranks = ranks[-out_size:]
P_out = P_world.create_subpartition(out_ranks)
P_out_cart = P_out.create_cartesian_subpartition(out_dims)

tensor_sizes = np.array([7, 5])
layer = transpose.DistributedTranspose(tensor_sizes, P_in_cart, P_out_cart)

f = transpose.DistributedTransposeFunction

if P_in_cart.active:
    x = np.arange(np.prod(tensor_sizes), dtype=np.float64).reshape(tensor_sizes)
    x = torch.from_numpy(x)

else:
    x = None

print_sequential(P_world.comm, f"rank = {P_world.rank}, x=\n{x}")

ctx = f()

x_scattered = f.forward(ctx, x, layer.P_common, layer.sizes,
                        layer.P_in, layer.in_data, layer.in_buffers,
                        layer.P_out, layer.out_data, layer.out_buffers)

print_sequential(P_world.comm, f"rank = {P_world.rank}, x_scattered =\n{x_scattered}")
