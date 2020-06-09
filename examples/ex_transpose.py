import numpy as np
import torch
from mpi4py import MPI

import distdl.utilities.slicing as slicing
from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.transpose import DistributedTranspose
from distdl.nn.transpose import DistributedTransposeFunction
from distdl.utilities.debug import print_sequential
from distdl.utilities.torch import NoneTensor

# Set up MPI cartesian communicator

P_world = MPIPartition(MPI.COMM_WORLD)
P_world.comm.Barrier()

in_dims = (4, 1)
out_dims = (2, 2)
in_size = np.prod(in_dims)
out_size = np.prod(out_dims)

P_in = P_world.create_partition_inclusive(np.arange(0, in_size))
PC_in = P_in.create_cartesian_topology_partition(in_dims)

P_out = P_world.create_partition_inclusive(np.arange(P_world.size-out_size, P_world.size))
PC_out = P_out.create_cartesian_topology_partition(out_dims)

tensor_sizes = np.array([7, 5])

layer = DistributedTranspose(tensor_sizes, PC_in, PC_out)

if PC_in.active:
    in_subsizes = slicing.compute_subsizes(PC_in.comm.dims,
                                           PC_in.comm.Get_coords(P_in.rank),
                                           tensor_sizes)
    x = np.zeros(in_subsizes) + P_in.rank + 1
    x = torch.from_numpy(x)
else:
    x = NoneTensor()

print_sequential(P_world.comm, f"x_{P_world.rank}: {x}")

ctx = DistributedTransposeFunction()

y = DistributedTransposeFunction.forward(ctx, x,
                                         layer.P_union,
                                         layer.global_tensor_sizes,
                                         layer.P_in,
                                         layer.in_data,
                                         layer.in_buffers,
                                         layer.P_out,
                                         layer.out_data,
                                         layer.out_buffers,
                                         layer.dtype)
print_sequential(P_world.comm, f"y_{P_world.rank}: {y}")

if PC_out.active:
    out_subsizes = slicing.compute_subsizes(PC_out.comm.dims,
                                            PC_out.comm.Get_coords(P_out.rank),
                                            tensor_sizes)
    gy = np.zeros(out_subsizes) + P_out.rank + 1
    gy = torch.from_numpy(gy)
else:
    gy = NoneTensor()

print_sequential(P_world.comm, f"gy_{P_world.rank}: {gy}")

gx = DistributedTransposeFunction.backward(ctx, gy)
print_sequential(P_world.comm, f"gx_{P_world.rank}: {gx}")
