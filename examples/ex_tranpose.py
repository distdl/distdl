import numpy as np
import torch
from mpi4py import MPI

import distdl.nn.transpose as transpose
import distdl.utilities.slicing as slicing
from distdl.backends.mpi.partition import MPIPartition
from distdl.utilities.debug import print_sequential
from distdl.utilities.misc import DummyContext

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
ranks = np.arange(P_world.size)

in_dims = (4, 1)
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
    in_subsizes = slicing.compute_subsizes(P_in_cart.comm.dims,
                                           P_in_cart.comm.Get_coords(P_in.rank),
                                           tensor_sizes)
    x = np.zeros(in_subsizes) + P_in.rank + 1
    x = torch.from_numpy(x)
else:
    x = None

print_sequential(P_world.comm, f"x_{P_world.rank}: {x}")

ctx = DummyContext()

y = f.forward(ctx, x, layer.P_common, layer.sizes,
              layer.P_in, layer.in_data, layer.in_buffers,
              layer.P_out, layer.out_data, layer.out_buffers)
print_sequential(P_world.comm, f"y_{P_world.rank}: {y}")

if P_out_cart.active:
    out_subsizes = slicing.compute_subsizes(P_out_cart.comm.dims,
                                            P_out_cart.comm.Get_coords(P_out.rank),
                                            tensor_sizes)
    gy = np.zeros(out_subsizes) + P_out.rank + 1
    gy = torch.from_numpy(gy)
else:
    gy = None

print_sequential(P_world.comm, f"gy_{P_world.rank}: {gy}")

gx = f.backward(ctx, gy)
print_sequential(P_world.comm, f"gx_{P_world.rank}: {gx}")
