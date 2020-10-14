import numpy as np
import pytest
import torch
from adjoint_test import check_adjoint_test_tight

adjoint_parametrizations = []

# Main functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(4, 8), [4, 1],  # P_x_ranks, P_x_shape
        np.arange(0, 12), [3, 4],  # P_y_ranks, P_y_shape
        [77, 55],  # x_global_shape
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-overlap-3D",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [4, 1],  # P_x_ranks, P_x_shape
        np.arange(4, 16), [3, 4],  # P_y_ranks, P_y_shape
        [77, 55],  # x_global_shape
        16,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-disjoint-3D",
        marks=[pytest.mark.mpi(min_size=16)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [4, 1],  # P_x_ranks, P_x_shape
        np.arange(5, 17), [3, 4],  # P_y_ranks, P_y_shape
        [77, 55],  # x_global_shape
        17,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-disjoint-inactive-3D",
        marks=[pytest.mark.mpi(min_size=17)]
        )
    )

# Sequential functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 1), [1, 1],  # P_x_ranks, P_x_shape
        np.arange(0, 1), [1, 1],  # P_y_ranks, P_y_shape
        [77, 55],  # x_global_shape
        1,  # passed to comm_split_fixture, required MPI ranks
        id="sequential-identity",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

# As a scatter
adjoint_parametrizations.append(
    pytest.param(
        np.arange(4, 5), [1, 1],  # P_x_ranks, P_x_shape
        np.arange(0, 12), [3, 4],  # P_y_ranks, P_y_shape
        [77, 55],  # x_global_shape
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-as_scatter-overlap-3D",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 1), [1, 1],  # P_x_ranks, P_x_shape
        np.arange(1, 13), [3, 4],  # P_y_ranks, P_y_shape
        [77, 55],  # x_global_shape
        13,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-as_scatter-disjoint-3D",
        marks=[pytest.mark.mpi(min_size=13)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 1), [1, 1],  # P_x_ranks, P_x_shape
        np.arange(2, 14), [3, 4],  # P_y_ranks, P_y_shape
        [77, 55],  # x_global_shape
        14,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-as_scatter-disjoint-inactive-3D",
        marks=[pytest.mark.mpi(min_size=14)]
        )
    )

# As a gather
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [3, 4],  # P_x_ranks, P_x_shape
        np.arange(4, 5), [1, 1],  # P_y_ranks, P_y_shape
        [77, 55],  # x_global_shape
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-as_gather-overlap-3D",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(1, 13), [3, 4],  # P_x_ranks, P_x_shape
        np.arange(0, 1), [1, 1],  # P_y_ranks, P_y_shape
        [77, 55],  # x_global_shape
        13,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-as_gather-disjoint-3D",
        marks=[pytest.mark.mpi(min_size=13)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(2, 14), [3, 4],  # P_x_ranks, P_x_shape
        np.arange(0, 1), [1, 1],  # P_y_ranks, P_y_shape
        [77, 55],  # x_global_shape
        14,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-as_gather-disjoint-inactive-3D",
        marks=[pytest.mark.mpi(min_size=14)]
        )
    )


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "P_y_ranks, P_y_shape,"
                         "x_global_shape,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
@pytest.mark.parametrize("balanced", [True, False])
def test_transpose_adjoint(barrier_fence_fixture,
                           comm_split_fixture,
                           P_x_ranks, P_x_shape,
                           P_y_ranks, P_y_shape,
                           x_global_shape,
                           balanced):

    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.utilities.slicing import compute_subshape
    from distdl.utilities.torch import zero_volume_tensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    P_y_base = P_world.create_partition_inclusive(P_y_ranks)
    P_y = P_y_base.create_cartesian_topology_partition(P_y_shape)

    # The global tensor size is the same for x and y
    layer = DistributedTranspose(P_x, P_y, preserve_batch=False)

    # Forward Input
    x = zero_volume_tensor()
    if P_x.active:
        if balanced:
            x_local_shape = compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
        else:
            quotient = np.atleast_1d(x_global_shape) // np.atleast_1d(P_x_shape)
            remainder = np.atleast_1d(x_global_shape) % np.atleast_1d(P_x_shape)
            loc = np.where(P_x.index == 0)
            x_local_shape = quotient.copy()
            x_local_shape[loc] += remainder[loc]

        x = torch.randn(*x_local_shape)

    x.requires_grad = True

    # Adjoint Input
    dy = zero_volume_tensor()
    if P_y.active:
        y_local_shape = compute_subshape(P_y.shape,
                                         P_y.index,
                                         x_global_shape)
        dy = torch.randn(*y_local_shape)

    # y = F @ x
    y = layer(x)

    # dx = F* @ dy
    y.backward(dy)
    dx = x.grad

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, x, dx, y, dy)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
    P_y_base.deactivate()
    P_y.deactivate()


@pytest.mark.parametrize("comm_split_fixture", [4], indirect=["comm_split_fixture"])
def test_excepts_mismatched_partitions(barrier_fence_fixture,
                                       comm_split_fixture):

    import numpy as np

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    in_shape = (1, 4, 1, 1)
    out_shape = (1, 2)

    in_size = np.prod(in_shape)
    out_size = np.prod(out_shape)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(np.arange(0, in_size))
    P_x = P_x_base.create_cartesian_topology_partition(in_shape)

    P_y_base = P_world.create_partition_inclusive(np.arange(P_world.size-out_size, P_world.size))
    P_y = P_y_base.create_cartesian_topology_partition(out_shape)

    with pytest.raises(ValueError) as e_info:  # noqa: F841
        DistributedTranspose(P_x, P_y)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
    P_y_base.deactivate()
    P_y.deactivate()


@pytest.mark.parametrize("comm_split_fixture", [4], indirect=["comm_split_fixture"])
def test_excepts_mismatched_input_partition_tensor(barrier_fence_fixture,
                                                   comm_split_fixture):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.utilities.slicing import compute_subshape
    from distdl.utilities.torch import zero_volume_tensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Input partition rank must match tensor rank
    in_shape = (1, 4, 1, 1)
    out_shape = (1, 1, 1, 2)
    x_global_shape = np.array([16, 5, 5])

    in_size = np.prod(in_shape)
    out_size = np.prod(out_shape)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(np.arange(0, in_size))
    P_x = P_x_base.create_cartesian_topology_partition(in_shape)

    P_y_base = P_world.create_partition_inclusive(np.arange(P_world.size-out_size, P_world.size))
    P_y = P_y_base.create_cartesian_topology_partition(out_shape)

    with pytest.raises(ValueError) as e_info:  # noqa: F841
        layer = DistributedTranspose(P_x, P_y)

        # Forward Input
        x = zero_volume_tensor()
        if P_x.active:
            x_local_shape = compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
            x = torch.randn(*x_local_shape)
        x.requires_grad = True

        layer(x)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
    P_y_base.deactivate()
    P_y.deactivate()


@pytest.mark.parametrize("comm_split_fixture", [4], indirect=["comm_split_fixture"])
def test_excepts_mismatched_output_partition_tensor(barrier_fence_fixture,
                                                    comm_split_fixture):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.utilities.slicing import compute_subshape
    from distdl.utilities.torch import zero_volume_tensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Output partition rank must match tensor rank
    in_shape = (4, 1, 1)
    out_shape = (1, 1, 1, 2)
    x_global_shape = np.array([16, 5, 5])

    in_size = np.prod(in_shape)
    out_size = np.prod(out_shape)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(np.arange(0, in_size))
    P_x = P_x_base.create_cartesian_topology_partition(in_shape)

    P_y_base = P_world.create_partition_inclusive(np.arange(P_world.size-out_size, P_world.size))
    P_y = P_y_base.create_cartesian_topology_partition(out_shape)

    with pytest.raises(ValueError) as e_info:  # noqa: F841
        layer = DistributedTranspose(P_x, P_y)

        # Forward Input
        x = zero_volume_tensor()
        if P_x.active:
            x_local_shape = compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
            x = torch.randn(*x_local_shape)
        x.requires_grad = True

        layer(x)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
    P_y_base.deactivate()
    P_y.deactivate()


@pytest.mark.parametrize("comm_split_fixture", [4], indirect=["comm_split_fixture"])
def test_excepts_mismatched_nondivisible_tensor(barrier_fence_fixture,
                                                comm_split_fixture):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.utilities.slicing import compute_subshape
    from distdl.utilities.torch import zero_volume_tensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # A tensor with size 1 in a dimension cannot be partitioned in that
    # dimension.  (See last dimension of output and tensor.)
    in_shape = (1, 4, 1, 1)
    out_shape = (1, 1, 1, 2)
    x_global_shape = np.array([1, 16, 5, 1])

    in_size = np.prod(in_shape)
    out_size = np.prod(out_shape)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(np.arange(0, in_size))
    P_x = P_x_base.create_cartesian_topology_partition(in_shape)

    P_y_base = P_world.create_partition_inclusive(np.arange(P_world.size-out_size, P_world.size))
    P_y = P_y_base.create_cartesian_topology_partition(out_shape)

    with pytest.raises(ValueError) as e_info:  # noqa: F841
        layer = DistributedTranspose(P_x, P_y)

        # Forward Input
        x = zero_volume_tensor()
        if P_x.active:
            x_local_shape = compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
            x = torch.randn(*x_local_shape)
        x.requires_grad = True

        layer(x)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
    P_y_base.deactivate()
    P_y.deactivate()


dtype_parametrizations = []


# Main functionality
dtype_parametrizations.append(
    pytest.param(
        torch.float32, True,  # dtype, test_backward,
        np.arange(0, 4), [4, 1],  # P_x_ranks, P_x_shape
        np.arange(0, 4), [2, 2],  # P_y_ranks, P_y_shape
        [77, 55],  # x_global_shape
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-dtype-float32",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# Test that it works with ints as well, can't compute gradient here
dtype_parametrizations.append(
    pytest.param(
        torch.int32, False,  # dtype, test_backward,
        np.arange(0, 4), [4, 1],  # P_x_ranks, P_x_shape
        np.arange(0, 4), [2, 2],  # P_y_ranks, P_y_shape
        [77, 55],  # x_global_shape
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-dtype-int32",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# Also test doubles
dtype_parametrizations.append(
    pytest.param(
        torch.float64, True,  # dtype, test_backward,
        np.arange(0, 4), [4, 1],  # P_x_ranks, P_x_shape
        np.arange(0, 4), [2, 2],  # P_y_ranks, P_y_shape
        [77, 55],  # x_global_shape
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-dtype-float64",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("dtype, test_backward,"
                         "P_x_ranks, P_x_shape,"
                         "P_y_ranks, P_y_shape,"
                         "x_global_shape,"
                         "comm_split_fixture",
                         dtype_parametrizations,
                         indirect=["comm_split_fixture"])
def test_transpose_dtype(barrier_fence_fixture,
                         comm_split_fixture,
                         dtype, test_backward,
                         P_x_ranks, P_x_shape,
                         P_y_ranks, P_y_shape,
                         x_global_shape):

    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.utilities.slicing import compute_subshape
    from distdl.utilities.torch import zero_volume_tensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    P_y_base = P_world.create_partition_inclusive(P_y_ranks)
    P_y = P_y_base.create_cartesian_topology_partition(P_y_shape)

    # The global tensor size is the same for x and y
    layer = DistributedTranspose(P_x, P_y, preserve_batch=False)

    # Forward Input
    x = zero_volume_tensor(dtype=dtype)
    if P_x.active:
        x_local_shape = compute_subshape(P_x.shape,
                                         P_x.index,
                                         x_global_shape)
        x = 10*torch.randn(*x_local_shape).to(dtype)

    x.requires_grad = test_backward
    # y = F @ x
    y = layer(x)
    if P_y.active:
        assert y.dtype == dtype

    if test_backward:
        # Adjoint Input
        dy = zero_volume_tensor(dtype=dtype)
        if P_y.active:
            y_local_shape = compute_subshape(P_y.shape,
                                             P_y.index,
                                             x_global_shape)
            dy = 10*torch.randn(*y_local_shape).to(dtype)

        # dx = F* @ dy
        y.backward(dy)
        dx = x.grad
        if P_x.active:
            assert dx.dtype == dtype

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
    P_y_base.deactivate()
    P_y.deactivate()


identity_parametrizations = []

# Main functionality
identity_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [3, 4],  # P_x_ranks, P_x_shape
        [77, 55],  # x_global_shape
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-identity-2D",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

identity_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 4],  # P_x_ranks, P_x_shape
        [77, 55],  # x_global_shape
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-identity-1D",
        marks=[pytest.mark.mpi(min_size=16)]
        )
    )


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "x_global_shape,"
                         "comm_split_fixture",
                         identity_parametrizations,
                         indirect=["comm_split_fixture"])
@pytest.mark.parametrize("balanced", [True, False])
def test_transpose_identity(barrier_fence_fixture,
                            comm_split_fixture,
                            P_x_ranks, P_x_shape,
                            x_global_shape,
                            balanced):

    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.utilities.slicing import compute_subshape
    from distdl.utilities.torch import zero_volume_tensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    P_y = P_x

    # The global tensor size is the same for x and y
    layer = DistributedTranspose(P_x, P_y, preserve_batch=False)

    # Forward Input
    x = zero_volume_tensor()
    if P_x.active:
        if balanced:
            x_local_shape = compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
        else:
            quotient = np.atleast_1d(x_global_shape) // np.atleast_1d(P_x_shape)
            remainder = np.atleast_1d(x_global_shape) % np.atleast_1d(P_x_shape)
            loc = np.where(P_x.index == 0)
            x_local_shape = quotient.copy()
            x_local_shape[loc] += remainder[loc]

        x = torch.randn(*x_local_shape)

    x.requires_grad = True

    # Adjoint Input
    dy = zero_volume_tensor()
    if P_y.active:
        y_local_shape = compute_subshape(P_y.shape,
                                         P_y.index,
                                         x_global_shape)
        dy = torch.randn(*y_local_shape)

    # y = F @ x
    y = layer(x)

    # In the balanced case, this should be a true identity, so there should
    # be no communication performed, just self-copies.
    if balanced:
        for sl, sz, p in layer.P_x_to_y_overlaps:
            assert p == "self" or (sl, sz, p) == (None, None, None)
        for sl, sz, p in layer.P_y_to_x_overlaps:
            assert p == "self" or (sl, sz, p) == (None, None, None)

    # dx = F* @ dy
    y.backward(dy)
    dx = x.grad

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, x, dx, y, dy)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
