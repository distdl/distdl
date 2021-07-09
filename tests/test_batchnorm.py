import numpy as np
import pytest
import torch

import distdl
from distdl.utilities.torch import zero_volume_tensor

ERROR_THRESHOLD = 1e-4
parametrizations_affine = []

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [2, 1, 2],  # P_x_ranks, P_x_shape,
        (4, 3, 10),  # input_shape
        3, 1e-05, 0.1, True,  # num_features, eps, momentum, affine,
        False,  # track_running_statistics
        [0],  # affine_workers
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-batch",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [1, 4, 1],  # P_x_ranks, P_x_shape,
        (4, 4, 10),  # input_shape
        4, 1e-03, 0.01, True,  # num_features, eps, momentum, affine,
        True,  # track_running_statistics
        [0, 1, 2, 3],  # affine_workers
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-feature",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [1, 2, 2],  # P_x_ranks, P_x_shape,
        (7, 13, 11),  # input_shape
        13, 1e-05, 0.01, True,  # num_features, eps, momentum, affine,
        True,  # track_running_statistics
        [0, 2],  # affine_workers
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-feature-track-running",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 4), [2, 2],  # P_x_ranks, P_x_shape,
        (7, 13),  # input_shape
        13, 1e-05, 0.1, True,  # num_features, eps, momentum, affine,
        False,  # track_running_statistics
        [0, 1],  # affine_workers
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-2d",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 8), [1, 2, 2, 2],  # P_x_ranks, P_x_shape,
        (7, 13, 11, 3),  # input_shape
        13, 1e-05, 0.1, True,  # num_features, eps, momentum, affine,
        False,  # track_running_statistics
        [0, 4],  # affine_workers
        8,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-4d",
        marks=[pytest.mark.mpi(min_size=8)]
        )
    )

parametrizations_affine.append(
    pytest.param(
        np.arange(0, 12), [1, 3, 2, 2],  # P_x_ranks, P_x_shape,
        (7, 13, 11, 3),  # input_shape
        13, 1e-05, 0.1, True,  # num_features, eps, momentum, affine,
        False,  # track_running_statistics
        [0, 4, 8],  # affine_workers
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-affine-4d-many-ranks",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

parametrizations_non_affine = []

parametrizations_non_affine.append(
    pytest.param(
        np.arange(0, 4), [1, 2, 2],  # P_x_ranks, P_x_shape,
        (7, 13, 11),  # input_shape
        13, 1e-05, 0.1, False,  # num_features, eps, momentum, affine,
        False,  # track_running_statistics
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-no-affine-feature",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations_non_affine.append(
    pytest.param(
        np.arange(0, 4), [1, 2, 2],  # P_x_ranks, P_x_shape,
        (7, 13, 11),  # input_shape
        13, 1e-05, 0.1, False,  # num_features, eps, momentum, affine,
        True,  # track_running_statistics
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-batch-norm-no-affine-feature-track-running",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )


@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "input_shape,"
                         "num_features, eps, momentum, affine,"
                         "track_running_stats,"
                         "affine_workers,"
                         "comm_split_fixture",
                         parametrizations_affine,
                         indirect=["comm_split_fixture"])
def test_batch_norm_with_training(barrier_fence_fixture,
                                  P_x_ranks, P_x_shape,
                                  input_shape,
                                  num_features, eps, momentum, affine,
                                  track_running_stats,
                                  affine_workers,
                                  comm_split_fixture):

    from distdl.backends.mpi.partition import MPIPartition

    torch.manual_seed(0)

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    num_dimensions = len(input_shape)
    P_in_out_base = P_world.create_partition_inclusive([0])
    P_in_out = P_in_out_base.create_cartesian_topology_partition([1] * num_dimensions)
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)
    P_affine = P_world.create_partition_inclusive(affine_workers)

    # Create the input
    if P_world.rank == 0:
        input_train = torch.rand(input_shape, dtype=torch.float32)
        input_eval = torch.rand(input_shape, dtype=torch.float32)
        exp = torch.rand(input_shape, dtype=torch.float32)
    else:
        input_train = zero_volume_tensor()
        input_eval = zero_volume_tensor()
        exp = zero_volume_tensor()

    # Create the sequential network
    if len(input_shape) == 2:
        seq_layer = torch.nn.BatchNorm1d
    elif len(input_shape) == 3:
        seq_layer = torch.nn.BatchNorm1d
    elif len(input_shape) == 4:
        seq_layer = torch.nn.BatchNorm2d
    elif len(input_shape) == 5:
        seq_layer = torch.nn.BatchNorm3d
    if P_world.rank == 0:
        seq_bn = seq_layer(num_features=num_features,
                           eps=eps,
                           momentum=momentum,
                           affine=affine,
                           track_running_stats=track_running_stats)

    # Train sequential network
    if P_world.rank == 0:
        seq_bn.train()
        seq_out1 = seq_bn(input_train)
        seq_loss = ((seq_out1 - exp)**2).sum()
        seq_loss.backward()
        seq_grads = [p.grad for p in seq_bn.parameters()]
        # Do a manual weight update (this is what optimizer does):
        with torch.no_grad():
            for p in seq_bn.parameters():
                p.copy_(p + 0.1*p.grad)

    # Evaluate sequential network
    if P_world.rank == 0:
        seq_bn.eval()
        seq_out2 = seq_bn(input_eval)

    # Create distributed network
    tr1 = distdl.nn.DistributedTranspose(P_in_out, P_x)
    dist_bn = distdl.nn.DistributedBatchNorm(P_x,
                                             num_features=num_features,
                                             eps=eps,
                                             momentum=momentum,
                                             affine=affine,
                                             track_running_stats=track_running_stats)
    tr2 = distdl.nn.DistributedTranspose(P_x, P_in_out)

    # Only rank 0 should have trainable parameters:
    if P_world.rank in affine_workers:
        assert len(list(dist_bn.parameters())) == 2
    else:
        assert len(list(dist_bn.parameters())) == 0

    # Train distributed network
    dist_bn.train()
    dist_out1 = tr2(dist_bn(tr1(input_train)))
    dist_loss = ((dist_out1 - exp)**2).sum()
    assert dist_loss.requires_grad
    dist_loss.backward()
    # Note: We expect the batch norm gradient to have extra dimensions than PyTorch,
    #       but both ultimately have volume equal to num_features.
    #       So, reshape them now, and gather them onto rank 0 for comparison.
    if P_world.rank == 0:
        dist_grads = []
    if P_world.rank in affine_workers:
        for p in dist_bn.parameters():
            if affine_workers == [0]:
                parts = [p.grad]
            else:
                parts = P_affine._comm.gather(p.grad, root=0)
            if P_world.rank == 0:
                grad = torch.cat(parts, 1)
                reshaped = grad.reshape((num_features,))
                dist_grads.append(reshaped)
    # Do a manual weight update (this is what optimizer does):
    with torch.no_grad():
        for p in dist_bn.parameters():
            p.copy_(p + 0.1*p.grad)

    # Evaluate distributed network
    dist_bn.eval()
    dist_out2 = tr2(dist_bn(tr1(input_eval)))

    # Compare the distributed and sequential networks
    if P_world.rank == 0:

        # Set the absolute tolerance to ~sqrt(e_mach), or the default
        # Pytorch got their defaults from NumPy, but NumPy defaults to 64-bit
        # floats, not 32-bit floats as torch does.  Consequently, the default
        # torch atol is actually tighter than one can expect from two fp-equal
        # floating point numbers.  The NumPy default of 1e-8 is closer to
        # sqrt(e_mach) for 64-bit numbers.  So we set the 32-bit tolerance to
        # sqrt(1e-7), as our usual choice, 1e-5, is too tight.
        if seq_out1.dtype == torch.float64:
            atol = 1e-8
        elif seq_out1.dtype == torch.float32:
            import math
            atol = math.sqrt(1e-7)
        else:
            # torch default
            atol = 1e-8

        assert dist_out1.shape == seq_out1.shape
        assert torch.allclose(dist_out1, seq_out1, rtol=ERROR_THRESHOLD, atol=atol)
        assert dist_loss.shape == seq_loss.shape
        assert torch.allclose(dist_loss, seq_loss, rtol=ERROR_THRESHOLD, atol=atol)
        for dist_grad, seq_grad in zip(dist_grads, seq_grads):
            assert dist_grad.shape == seq_grad.shape
            assert torch.allclose(dist_grad, seq_grad, rtol=ERROR_THRESHOLD, atol=atol)
        assert dist_out2.shape == seq_out2.shape
        assert torch.allclose(dist_out2, seq_out2, rtol=ERROR_THRESHOLD, atol=atol)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
    P_in_out_base.deactivate()
    P_in_out.deactivate()
    P_affine.deactivate()


@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "input_shape,"
                         "num_features, eps, momentum, affine,"
                         "track_running_stats,"
                         "comm_split_fixture",
                         parametrizations_non_affine,
                         indirect=["comm_split_fixture"])
def test_batch_norm_no_training(barrier_fence_fixture,
                                P_x_ranks, P_x_shape,
                                input_shape,
                                num_features, eps, momentum, affine,
                                track_running_stats,
                                comm_split_fixture):

    from distdl.backends.mpi.partition import MPIPartition

    torch.manual_seed(0)

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    num_dimensions = len(input_shape)
    P_in_out_base = P_world.create_partition_inclusive([0])
    P_in_out = P_in_out_base.create_cartesian_topology_partition([1] * num_dimensions)
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    # Create the input
    if P_world.rank == 0:
        input_eval = torch.rand(input_shape, dtype=torch.float32)
    else:
        input_eval = zero_volume_tensor()

    # Create the sequential network
    if P_world.rank == 0:
        seq_net = torch.nn.Sequential(torch.nn.BatchNorm1d(num_features=num_features,
                                                           eps=eps,
                                                           momentum=momentum,
                                                           affine=affine,
                                                           track_running_stats=track_running_stats))

    # Evaluate sequential network
    if P_world.rank == 0:
        seq_net.eval()
        seq_out = seq_net(input_eval)

    # Create distributed network
    dist_net = torch.nn.Sequential(distdl.nn.DistributedTranspose(P_in_out, P_x),
                                   distdl.nn.DistributedBatchNorm(P_x,
                                                                  num_features=num_features,
                                                                  eps=eps,
                                                                  momentum=momentum,
                                                                  affine=affine,
                                                                  track_running_stats=track_running_stats),
                                   distdl.nn.DistributedTranspose(P_x, P_in_out))

    # Evaluate distributed network
    dist_net.eval()
    dist_out = dist_net(input_eval)

    # Compare the distributed and sequential networks
    if P_world.rank == 0:
        assert dist_out.shape == seq_out.shape

        # Set the absolute tolerance to ~sqrt(e_mach), or the default
        # Pytorch got their defaults from NumPy, but NumPy defaults to 64-bit
        # floats, not 32-bit floats as torch does.  Consequently, the default
        # torch atol is actually tighter than one can expect from two fp-equal
        # floating point numbers.  The NumPy default of 1e-8 is closer to
        # sqrt(e_mach) for 64-bit numbers.  So we set the 32-bit tolerance to
        # a little tighter than sqrt(1e-7), 1e-5.
        if seq_out.dtype == torch.float64:
            atol = 1e-8
        elif seq_out.dtype == torch.float32:
            atol = 1e-5
        else:
            # torch default
            atol = 1e-8
        assert torch.allclose(dist_out, seq_out, atol=atol)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
    P_in_out_base.deactivate()
    P_in_out.deactivate()
