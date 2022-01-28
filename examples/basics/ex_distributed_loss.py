# This example demonstrates the behavior of distributed loss functions.
#
# It requires 4 workers to run.
#
# Run with, e.g.,
#     > mpirun -np 4 python ex_distributed_loss.py

import warnings

import numpy as np
import torch
from mpi4py import MPI

import distdl
from distdl.backends.mpi.partition import MPIPartition
from distdl.nn import Repartition
from distdl.utilities.torch import zero_volume_tensor

warnings.filterwarnings("ignore", category=UserWarning)

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Create the input/output partition (using the first 4 workers)
in_shape = (4, )
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)
P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Also extract the first worker to do a local computation
P_0_base = P_world.create_partition_inclusive([0])
P_0 = P_0_base.create_cartesian_topology_partition([1])

# Setup the loss function arguments.  Other valid configurations are commented
# out.
loss_args = {"reduction": "mean"}
# loss_args = {"reduction": "sum"}
# loss_args = {"reduction": "none"}
# loss_args = {"reduction": "batchmean"}  # Valid for KL-Divergence losses.

# Loss, DistributedLoss = torch.nn.L1Loss, distdl.nn.DistributedL1Loss
Loss, DistributedLoss = torch.nn.MSELoss, distdl.nn.DistributedMSELoss
# Loss, DistributedLoss = torch.nn.PoissonNLLLoss, distdl.nn.DistributedPoissonNLLLoss
# Loss, DistributedLoss = torch.nn.GaussianNLLLoss, distdl.nn.DistributedGaussianNLLLoss
# Loss, DistributedLoss = torch.nn.BCELoss, distdl.nn.DistributedBCELoss
# Loss, DistributedLoss = torch.nn.BCEWithLogitsLoss, distdl.nn.DistributedBCEWithLogitsLoss
# Loss, DistributedLoss = torch.nn.KLDivLoss, distdl.nn.DistributedKLDivLoss

# Assemble scatter and gather layers so we can compare sequential and
# distributed losses.  A bug in Repartition (bug#182) requires two
# scatter layers here when 1 should suffice.
scatter = Repartition(P_0, P_x)
scatter2 = Repartition(P_0, P_x)
gather = Repartition(P_x, P_0)

# Setup the input and target tensors
global_tensor_size = [11]
with torch.no_grad():
    x = zero_volume_tensor()
    y = zero_volume_tensor()
    if P_0.active:
        x = torch.zeros(global_tensor_size)
        y = torch.randn(global_tensor_size)
        x.requires_grad = True

    x_l = scatter(x)
    y_l = scatter2(y)
    x_l.requires_grad = True


# Do the sequential loss
if P_0.active:
    seq_criterion = Loss(**loss_args)
    seq_loss = seq_criterion(x, y)

# Do the distributed loss
dist_criterion = DistributedLoss(P_x, **loss_args)
dist_loss = dist_criterion(x_l, y_l)

# Compare the outputs.  If there is no reduction, we cannot compare the
# adjoint application but we can gather the loss tensor and compare that.
# If there is a reduction, we can just apply the adjoint and compare both
# the reduced loss and the adjoint application.
if loss_args["reduction"] == "none":
    dist_loss = gather(dist_loss)

    if P_0.active:

        print(seq_loss, "\n", dist_loss)
        assert(torch.allclose(dist_loss, seq_loss))

else:
    dist_loss.backward()
    dx = gather(x_l.grad)

    if P_0.active:
        seq_loss.backward()

        print(seq_loss, dist_loss)
        print(x.grad.shape, "\n", x.grad)
        print(dx.shape, "\n", dx)

        assert(torch.allclose(dx, x.grad))
        assert(torch.allclose(dist_loss, seq_loss))
