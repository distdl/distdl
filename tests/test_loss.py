from mpi4py import MPI
import numpy as np
import torch

import distdl
from distdl.nn import DistributedTranspose
from distdl.backends.mpi.partition import MPIPartition
from distdl.utilities.torch import zero_volume_tensor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

P_world = MPIPartition(MPI.COMM_WORLD)

import math
P = math.sqrt(P_world.size/2)
K = 2
N = 2
B = 5

P_x = P_world.create_cartesian_topology_partition([2, P, P])
P_0 = P_world.create_partition_inclusive([0]).create_cartesian_topology_partition([1, 1, 1])


loss_args = {"reduction": "mean"}
# loss_args = {"reduction": "sum"}
# loss_args = {"reduction": "none"}
# loss_args = {"reduction": "batchmean"}

Loss, DistributedLoss = torch.nn.L1Loss, distdl.nn.DistributedL1Loss
Loss, DistributedLoss = torch.nn.MSELoss, distdl.nn.DistributedMSELoss
Loss, DistributedLoss = torch.nn.PoissonNLLLoss, distdl.nn.DistributedPoissonNLLLoss
# Loss, DistributedLoss = torch.nn.GaussianNLLLoss, distdl.nn.DistributedGaussianNLLLoss
Loss, DistributedLoss = torch.nn.BCELoss, distdl.nn.DistributedBCELoss
Loss, DistributedLoss = torch.nn.BCEWithLogitsLoss, distdl.nn.DistributedBCEWithLogitsLoss
# Loss, DistributedLoss = torch.nn.KLDivLoss, distdl.nn.DistributedKLDivLoss


scatter = DistributedTranspose(P_0, P_x)
gather = DistributedTranspose(P_x, P_0)

criterion = DistributedLoss(P_x, **loss_args)

with torch.no_grad():
    x = zero_volume_tensor()
    y = zero_volume_tensor()
    if P_0.active:
        x = torch.zeros((B, N, N))
        y = torch.randn((B, N, N))

    x_l = scatter(x)
    y_l = scatter(y)


x_l.requires_grad = True
global_loss = criterion(x_l, y_l)

if P_0.active:
    x.requires_grad = True
    crit = Loss(**loss_args)
    loss = crit(x, y)


if loss_args["reduction"] == "none":    
    dist_loss = gather(global_loss)

    if P_0.active:

        print(dist_loss, "\n", loss)
        assert(torch.allclose(dist_loss, loss))

else:
    global_loss.backward()
    dx = gather(x_l.grad)
    
    if P_0.active:
        loss.backward()

        print(loss, global_loss)
        print(x.grad.shape, "\n", x.grad)
        print(dx.shape, "\n", dx)

        assert(torch.allclose(dx, x.grad))
        assert(torch.allclose(global_loss, loss))
