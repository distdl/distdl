# This example demonstrates the behavior of the transpose primitive
# when used to gather data.
#
# It requires 4 workers to run.
#
# Run with, e.g.,
#     > mpirun -np 4 python ex_transpose_as_gather.py

import numpy as np
import torch
from mpi4py import MPI

import distdl.utilities.slicing as slicing
from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.repartition import Repartition
from distdl.utilities.torch import zero_volume_tensor

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Create the input partition (using the first 4 workers)
in_shape = (2, 2)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Create the output partition (using the last worker)
out_shape = (1, 1)
out_size = np.prod(out_shape)
out_workers = np.arange(P_world.size-out_size, P_world.size)

P_y_base = P_world.create_partition_inclusive(out_workers)
P_y = P_y_base.create_cartesian_topology_partition(out_shape)

# This global tensor shape is among the smallest useful shapes for an example
x_global_shape = np.array([7, 5])

# Create the transpose layer
layer = Repartition(P_x, P_y, preserve_batch=False)

# Setup the input tensor.  Any worker in P_x will generate its part of the
# input tensor.  Any worker not in P_x will have a zero-volume tensor.
#
# Input tensor will be (on a 2 x 2 partition):
# [ [ 1 1 1 | 2 2 ]
#   [ 1 1 1 | 2 2 ]
#   [ 1 1 1 | 2 2 ]
#   [ 1 1 1 | 2 2 ]
#   -------------
#   [ 3 3 3 | 4 4 ]
#   [ 3 3 3 | 4 4 ]
#   [ 3 3 3 | 4 4 ] ]
x = zero_volume_tensor()
if P_x.active:
    x_local_shape = slicing.compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
    x = np.zeros(x_local_shape) + P_x.rank + 1
    x = torch.from_numpy(x)
x.requires_grad = True
print(f"rank {P_world.rank}; index {P_x.index}; value {x}")

# Apply the layer.
#
# Output tensor will be (on a 1 x 1 partition):
# [ [ 1 1 1 2 2 ]
#   [ 1 1 1 2 2 ]
#   [ 1 1 1 2 2 ]
#   [ 1 1 1 2 2 ]
#   [ 3 3 3 4 4 ]
#   [ 3 3 3 4 4 ]
#   [ 3 3 3 4 4 ] ]

y = layer(x)
print(f"rank {P_world.rank}; index {P_y.index}; value {y}")

# Setup the adjoint input tensor.  Any worker in P_y will generate its part of
# the adjoint input tensor.  Any worker not in P_y will have a zero-volume
# tensor.
#
# Adjoint input tensor will be (on a 1 x 1 partition):
# [ [ 1 1 1 1 1 ]
#   [ 1 1 1 1 1 ]
#   [ 1 1 1 1 1 ]
#   [ 1 1 1 1 1 ]
#   [ 1 1 1 1 1 ]
#   [ 1 1 1 1 1 ]
#   [ 1 1 1 1 1 ] ]
dy = zero_volume_tensor()
if P_y.active:
    y_local_shape = slicing.compute_subshape(P_y.shape,
                                             P_y.index,
                                             x_global_shape)
    dy = np.zeros(y_local_shape) + P_y.rank + 1
    dy = torch.from_numpy(dy)
print(f"rank {P_world.rank}; index {P_y.index}; value {dy}")

# Apply the adjoint of the layer.
#
# Adjoint output tensor will be (on a 2 x 2 partition):
# [ [ 1 1 1 | 1 1 ]
#   [ 1 1 1 | 1 1 ]
#   [ 1 1 1 | 1 1 ]
#   [ 1 1 1 | 1 1 ]
#   -------------
#   [ 1 1 1 | 1 1 ]
#   [ 1 1 1 | 1 1 ]
#   [ 1 1 1 | 1 1 ] ]
y.backward(dy)
dx = x.grad
print(f"rank {P_world.rank}; index {P_x.index}; value {dx}")
