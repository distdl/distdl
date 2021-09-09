# This example demonstrates the behavior of the all-reduce primitive.
#
# It requires 6 workers to run.
#
# Run with, e.g.,
#     > mpirun -np 6 python ex_all_reduce.py

import numpy as np
import torch
from mpi4py import MPI

import distdl.utilities.slicing as slicing
from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.all_sum_reduce import AllSumReduce
from distdl.utilities.torch import zero_volume_tensor

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Create the input/output partition (using the first worker)
in_shape = (2, 3)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# The all-reduce layer operates along partitiond dimensions, not tensor
# dimensions.  Thus, along the dimensions that the reduction applies, the
# subtensors all must be the same size.  Thus, this global shape is evenly
# divisible by the partition.  Later we will have an example for applying the
# reduction on the tensor itself.
x_global_shape = np.array([6, 6])

# Setup the input tensor.  Any worker in P_x will generate its part of the
# input tensor.  Any worker not in P_x will have a zero-volume tensor.
#
# Input tensor will be (on a 2 x 3 partition):
# [ [ 1 1 | 2 2 | 3 3 ]
#   [ 1 1 | 2 2 | 3 3 ]
#   [ 1 1 | 2 2 | 3 3 ]
#   -------------------
#   [ 4 4 | 5 5 | 6 6 ]
#   [ 4 4 | 5 5 | 6 6 ]
#   [ 4 4 | 5 5 | 6 6 ] ]
x = zero_volume_tensor()
if P_x.active:
    x_local_shape = slicing.compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
    x = np.zeros(x_local_shape) + P_x.rank + 1
    x = torch.from_numpy(x)
x.requires_grad = True
print(f"rank {P_world.rank}; index {P_x.index}; value {x}")

# Create the all-reduce layer.  Note, only one of the keep/reduce axes is
# required.  If they are both specified they must be mutually coherent.
# Commented out declarations are equivalent.  `axes_reduce` is equivalent
# to PyTorch's dimension argument in torch.sum().
#
# Here we reduce the columns (axis 1), along the rows.
all_reduce_cols = AllSumReduce(P_x, axes_reduce=(1,))
# all_reduce_cols = AllSumReduce(P_x, axes_keep=(0,))
# all_reduce_cols = AllSumReduce(P_x, axes_reduce=(1,), axes_keep=(0,))
#
# Output tensor will be (on a 2 x 3 partition):
# [ [  6  6 |  6  6 |  6  6 ]
#   [  6  6 |  6  6 |  6  6 ]
#   [  6  6 |  6  6 |  6  6 ]
#   -------------------------
#   [ 15 15 | 15 15 | 15 15 ]
#   [ 15 15 | 15 15 | 15 15 ]
#   [ 15 15 | 15 15 | 15 15 ] ]
y = all_reduce_cols(x)

print(f"rank {P_world.rank}; index {P_x.index}; value {y}")


# Here we reduce the rows (axis 0), along the columns.
all_reduce_rows = AllSumReduce(P_x, axes_reduce=(0,))
#
# Output tensor will be (on a 2 x 3 partition):
# [ [ 5 5 | 7 7 | 9 9 ]
#   [ 5 5 | 7 7 | 9 9 ]
#   [ 5 5 | 7 7 | 9 9 ]
#   -------------------
#   [ 5 5 | 7 7 | 9 9 ]
#   [ 5 5 | 7 7 | 9 9 ]
#   [ 5 5 | 7 7 | 9 9 ] ]
y = all_reduce_rows(x)

print(f"rank {P_world.rank}; index {P_x.index}; value {y}")


# Here we reduce all axes.
all_reduce_all = AllSumReduce(P_x, axes_reduce=(0, 1))
#
# Output tensor will be (on a 2 x 3 partition):
# [ [ 21 21 | 21 21 | 21 21 ]
#   [ 21 21 | 21 21 | 21 21 ]
#   [ 21 21 | 21 21 | 21 21 ]
#   -------------------------
#   [ 21 21 | 21 21 | 21 21 ]
#   [ 21 21 | 21 21 | 21 21 ]
#   [ 21 21 | 21 21 | 21 21 ] ]
y = all_reduce_all(x)

print(f"rank {P_world.rank}; index {P_x.index}; value {y}")


# Here we reduce none of the axes.
all_reduce_none = AllSumReduce(P_x, axes_reduce=tuple())
#
# Output tensor will be (on a 2 x 3 partition):
# [ [ 1 1 | 2 2 | 3 3 ]
#   [ 1 1 | 2 2 | 3 3 ]
#   [ 1 1 | 2 2 | 3 3 ]
#   -------------------
#   [ 4 4 | 5 5 | 6 6 ]
#   [ 4 4 | 5 5 | 6 6 ]
#   [ 4 4 | 5 5 | 6 6 ] ]
y = all_reduce_none(x)

print(f"rank {P_world.rank}; index {P_x.index}; value {y}")


# Reset the input so that we do not have equal shapes along the reducing
# dimensons.
x_global_shape = np.array([5, 7])

# Input tensor will be (on a 2 x 3 partition):
# [ [ 1 1 1 | 2 2 | 3 3 ]
#   [ 1 1 1 | 2 2 | 3 3 ]
#   [ 1 1 1 | 2 2 | 3 3 ]
#   -------------------
#   [ 4 4 4 | 5 5 | 6 6 ]
#   [ 4 4 4 | 5 5 | 6 6 ] ]
x = zero_volume_tensor()
if P_x.active:
    x_local_shape = slicing.compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
    x = np.zeros(x_local_shape) + P_x.rank + 1
    x = torch.from_numpy(x)
x.requires_grad = True
print(f"rank {P_world.rank}; index {P_x.index}; value {x}")

# We cannot reduce along the rows directly here, because e.g., Rank 0 and Rank
# 1 have different shaped subtensors.  But if we first reduce locally along
# the rows, then we can reduce *that* tensor because they both will have the
# same shape.  This is useful in normalization, where the averaging
# dimensions should match on all subtensors.

# Recall the layer definition.  `axes_reduce` matches the arguments to torch.sum
# all_reduce_cols = AllSumReduce(P_x, axes_reduce=(1,))
x_prime = torch.sum(x, (1,), keepdim=True)
# New tensor will be (on a 2 x 3 partition):
# [ [  3 |  4 |  6 ]
#   [  3 |  4 |  6 ]
#   [  3 |  4 |  6 ]
#   -------------------
#   [ 12 | 10 | 12 ]
#   [ 12 | 10 | 12 ] ]
print(f"rank {P_world.rank}; index {P_x.index}; value {x_prime}")

# Which we can now all-reduce to obtain:
# [ [ 13 | 13 | 13 ]
#   [ 13 | 13 | 13 ]
#   [ 13 | 13 | 13 ]
#   -------------------
#   [ 34 | 34 | 34 ]
#   [ 34 | 34 | 34 ] ]
y = all_reduce_cols(x_prime)

print(f"rank {P_world.rank}; index {P_x.index}; value {y}")
