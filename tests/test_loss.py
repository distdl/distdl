import numpy as np
import pytest
import torch

import distdl

input_parametrizations = []

input_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [4],  # P_x_ranks, P_x_shape
        [11],  # x_global_shape
        4,  # passed to comm_split_fixture, required MPI ranks
        id="1D",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

input_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 4],  # P_x_ranks, P_x_shape
        [7, 7],  # x_global_shape
        4,  # passed to comm_split_fixture, required MPI ranks
        id="1D-nonpartitioned_batch",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

input_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [2, 2],  # P_x_ranks, P_x_shape
        [7, 7],  # x_global_shape
        4,  # passed to comm_split_fixture, required MPI ranks
        id="1D-partitioned_batch",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

input_parametrizations.append(
    pytest.param(
        np.arange(0, 8), [1, 2, 4],  # P_x_ranks, P_x_shape
        [5, 7, 11],  # x_global_shape
        8,  # passed to comm_split_fixture, required MPI ranks
        id="2D-nonpartitioned_batch",
        marks=[pytest.mark.mpi(min_size=8)]
        )
    )

input_parametrizations.append(
    pytest.param(
        np.arange(0, 8), [2, 2, 2],  # P_x_ranks, P_x_shape
        [5, 7, 11],  # x_global_shape
        8,  # passed to comm_split_fixture, required MPI ranks
        id="2D-partitioned_batch",
        marks=[pytest.mark.mpi(min_size=8)]
        )
    )

input_parametrizations.append(
    pytest.param(
        np.arange(0, 16), [1, 4, 2, 2],  # P_x_ranks, P_x_shape
        [9, 5, 7, 11],  # x_global_shape
        16,  # passed to comm_split_fixture, required MPI ranks
        id="3D-nonpartitioned_batch",
        marks=[pytest.mark.mpi(min_size=16)]
        )
    )

input_parametrizations.append(
    pytest.param(
        np.arange(0, 16), [2, 4, 2, 1],  # P_x_ranks, P_x_shape
        [9, 5, 7, 11],  # x_global_shape
        16,  # passed to comm_split_fixture, required MPI ranks
        id="3D-partitioned_batch",
        marks=[pytest.mark.mpi(min_size=16)]
        )
    )

loss_parametrizations = [
    # SequentialLoss, DistributedLoss
    pytest.param(torch.nn.L1Loss, distdl.nn.DistributedL1Loss),
    pytest.param(torch.nn.MSELoss, distdl.nn.DistributedMSELoss),
    pytest.param(torch.nn.PoissonNLLLoss, distdl.nn.DistributedPoissonNLLLoss),
    pytest.param(torch.nn.BCELoss, distdl.nn.DistributedBCELoss),
    pytest.param(torch.nn.BCEWithLogitsLoss, distdl.nn.DistributedBCEWithLogitsLoss),
    pytest.param(torch.nn.KLDivLoss, distdl.nn.DistributedKLDivLoss)
]


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "x_global_shape,"
                         "comm_split_fixture",
                         input_parametrizations,
                         indirect=["comm_split_fixture"])
@pytest.mark.parametrize("SequentialLoss,"
                         "DistributedLoss",
                         loss_parametrizations)
def test_distributed_loss(barrier_fence_fixture,
                          comm_split_fixture,
                          P_x_ranks, P_x_shape,
                          x_global_shape,
                          SequentialLoss,
                          DistributedLoss):

    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn import DistributedTranspose
    from distdl.utilities.torch import zero_volume_tensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    P_0_base = P_x_base.create_partition_inclusive([0])
    P_0 = P_0_base.create_cartesian_topology_partition([1]*len(P_x_shape))

    scatter = DistributedTranspose(P_0, P_x)
    gather = DistributedTranspose(P_x, P_0)

    for reduction in DistributedLoss._valid_reductions:

        distributed_criterion = DistributedLoss(P_x, reduction=reduction)
        sequential_criterion = SequentialLoss(reduction=reduction)

        with torch.no_grad():
            x_g = zero_volume_tensor()
            y_g = zero_volume_tensor()
            if P_0.active:
                x_g = torch.rand(x_global_shape)
                y_g = torch.rand(x_global_shape)

            x_l = scatter(x_g)
            y_l = scatter(y_g)

        x_l.requires_grad = True
        distributed_loss = distributed_criterion(x_l, y_l)

        # For "none", no reduction is applied so we see if it computed the
        # same loss as the sequential code by gathering the loss value it to
        # the root rank.
        if reduction == "none":
            distributed_loss = gather(distributed_loss)

        if P_0.active:
            x_g.requires_grad = True
            sequential_loss = sequential_criterion(x_g, y_g)

            assert(torch.allclose(distributed_loss, sequential_loss))

        # For any other reduction, we can compare the loss
        # value *and* backpropagate through the distributed loss to verify
        # that it produces the same output.
        if reduction != "none":
            distributed_loss.backward()
            distributed_dx_g = gather(x_l.grad)

            if P_0.active:
                sequential_loss.backward()
                sequential_dx_g = x_g.grad

                assert(torch.allclose(distributed_dx_g, sequential_dx_g))

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
    P_0_base.deactivate()
    P_0.deactivate()
