import numpy as np
import torch
from mpi4py import MPI
from torch.nn import Sequential

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.batch_norm import DistributedBatchNorm
from distdl.nn.transpose import DistributedTranspose
from distdl.utilities.debug import print_sequential
from distdl.utilities.torch import zero_volume_tensor

# Set up partitions
P_world = MPIPartition(MPI.COMM_WORLD)
P_base = P_world.create_partition_inclusive(np.arange(8))
P_in_out_base = P_world.create_partition_inclusive([0])
P_x = P_base.create_cartesian_topology_partition([4, 1, 2])
P_in_out = P_in_out_base.create_cartesian_topology_partition([1, 1, 1])

# Set a random input
if P_world.rank == 0:
    input = torch.rand([4, 3, 10], dtype=torch.float32)
else:
    input = zero_volume_tensor()

# Run the distributed bn layer
bn_net = Sequential(DistributedTranspose(P_in_out, P_x),
                    DistributedBatchNorm(P_x, num_features=3),
                    DistributedTranspose(P_x, P_in_out))
y = bn_net(input)

# Print the result
print_sequential(P_world._comm, f'output (rank {P_world.rank}): {y}')
