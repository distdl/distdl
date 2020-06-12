import numpy as np
import pytest
from adjoint_test import check_adjoint_test_tight

adjoint_parametrizations = []

# Main functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(4, 8), [4, 1],  # P_x_ranks, P_x_topo
        np.arange(0, 12), [3, 4],  # P_y_ranks, P_y_topo
        [77, 55],  # x_global_shape
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-overlap-3D",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [4, 1],  # P_x_ranks, P_x_topo
        np.arange(4, 16), [3, 4],  # P_y_ranks, P_y_topo
        [77, 55],  # x_global_shape
        16,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-disjoint-3D",
        marks=[pytest.mark.mpi(min_size=16)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 4), [4, 1],  # P_x_ranks, P_x_topo
        np.arange(5, 17), [3, 4],  # P_y_ranks, P_y_topo
        [77, 55],  # x_global_shape
        17,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-disjoint-inactive-3D",
        marks=[pytest.mark.mpi(min_size=17)]
        )
    )

# Sequential functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 1), [1, 1],  # P_x_ranks, P_x_topo
        np.arange(0, 1), [1, 1],  # P_y_ranks, P_y_topo
        [77, 55],  # x_global_shape
        1,  # passed to comm_split_fixture, required MPI ranks
        id="sequential-identity",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

# As a scatter
adjoint_parametrizations.append(
    pytest.param(
        np.arange(4, 5), [1, 1],  # P_x_ranks, P_x_topo
        np.arange(0, 12), [3, 4],  # P_y_ranks, P_y_topo
        [77, 55],  # x_global_shape
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-as_scatter-overlap-3D",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 1), [1, 1],  # P_x_ranks, P_x_topo
        np.arange(1, 13), [3, 4],  # P_y_ranks, P_y_topo
        [77, 55],  # x_global_shape
        13,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-as_scatter-disjoint-3D",
        marks=[pytest.mark.mpi(min_size=13)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 1), [1, 1],  # P_x_ranks, P_x_topo
        np.arange(2, 14), [3, 4],  # P_y_ranks, P_y_topo
        [77, 55],  # x_global_shape
        14,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-as_scatter-disjoint-inactive-3D",
        marks=[pytest.mark.mpi(min_size=14)]
        )
    )

# As a gather
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [3, 4],  # P_x_ranks, P_x_topo
        np.arange(4, 5), [1, 1],  # P_y_ranks, P_y_topo
        [77, 55],  # x_global_shape
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-as_gather-overlap-3D",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(1, 13), [3, 4],  # P_x_ranks, P_x_topo
        np.arange(0, 1), [1, 1],  # P_y_ranks, P_y_topo
        [77, 55],  # x_global_shape
        13,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-as_gather-disjoint-3D",
        marks=[pytest.mark.mpi(min_size=13)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(2, 14), [3, 4],  # P_x_ranks, P_x_topo
        np.arange(0, 1), [1, 1],  # P_y_ranks, P_y_topo
        [77, 55],  # x_global_shape
        14,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-as_gather-disjoint-inactive-3D",
        marks=[pytest.mark.mpi(min_size=14)]
        )
    )


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_topo,"
                         "P_y_ranks, P_y_topo,"
                         "x_global_shape,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_transpose_adjoint(barrier_fence_fixture,
                           comm_split_fixture,
                           P_x_ranks, P_x_topo,
                           P_y_ranks, P_y_topo,
                           x_global_shape):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_topo)

    P_y_base = P_world.create_partition_inclusive(P_y_ranks)
    P_y = P_y_base.create_cartesian_topology_partition(P_y_topo)

    # The global tensor size is the same for x and y
    layer = DistributedTranspose(P_x, P_y)

    # Forward Input
    x = NoneTensor()
    if P_x.active:
        x_local_shape = compute_subsizes(P_x.comm.dims,
                                         P_x.comm.Get_coords(P_x.rank),
                                         x_global_shape)
        x = torch.Tensor(np.random.randn(*x_local_shape))
    x.requires_grad = True

    # Adjoint Input
    dy = NoneTensor()
    if P_y.active:
        y_local_shape = compute_subsizes(P_y.comm.dims,
                                         P_y.comm.Get_coords(P_y.rank),
                                         x_global_shape)
        dy = torch.Tensor(np.random.randn(*y_local_shape))

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

    in_dims = (1, 4, 1, 1)
    out_dims = (1, 2)

    in_size = np.prod(in_dims)
    out_size = np.prod(out_dims)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(np.arange(0, in_size))
    P_x = P_x_base.create_cartesian_topology_partition(in_dims)

    P_y_base = P_world.create_partition_inclusive(np.arange(P_world.size-out_size, P_world.size))
    P_y = P_y_base.create_cartesian_topology_partition(out_dims)

    with pytest.raises(ValueError) as e_info:  # noqa: F841
        DistributedTranspose(P_x, P_y)


@pytest.mark.parametrize("comm_split_fixture", [4], indirect=["comm_split_fixture"])
def test_excepts_mismatched_input_partition_tensor(barrier_fence_fixture,
                                                   comm_split_fixture):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Input partition rank must match tensor rank
    in_dims = (1, 4, 1, 1)
    out_dims = (1, 1, 1, 2)
    x_global_shape = np.array([16, 5, 5])

    in_size = np.prod(in_dims)
    out_size = np.prod(out_dims)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(np.arange(0, in_size))
    P_x = P_x_base.create_cartesian_topology_partition(in_dims)

    P_y_base = P_world.create_partition_inclusive(np.arange(P_world.size-out_size, P_world.size))
    P_y = P_y_base.create_cartesian_topology_partition(out_dims)

    with pytest.raises(ValueError) as e_info:  # noqa: F841
        layer = DistributedTranspose(P_x, P_y)

        # Forward Input
        x = NoneTensor()
        if P_x.active:
            x_local_shape = compute_subsizes(P_x.comm.dims,
                                             P_x.comm.Get_coords(P_x.rank),
                                             x_global_shape)
            x = torch.Tensor(np.random.randn(*x_local_shape))
        x.requires_grad = True

        layer(x)


@pytest.mark.parametrize("comm_split_fixture", [4], indirect=["comm_split_fixture"])
def test_excepts_mismatched_output_partition_tensor(barrier_fence_fixture,
                                                    comm_split_fixture):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Output partition rank must match tensor rank
    in_dims = (4, 1, 1)
    out_dims = (1, 1, 1, 2)
    x_global_shape = np.array([16, 5, 5])

    in_size = np.prod(in_dims)
    out_size = np.prod(out_dims)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(np.arange(0, in_size))
    P_x = P_x_base.create_cartesian_topology_partition(in_dims)

    P_y_base = P_world.create_partition_inclusive(np.arange(P_world.size-out_size, P_world.size))
    P_y = P_y_base.create_cartesian_topology_partition(out_dims)

    with pytest.raises(ValueError) as e_info:  # noqa: F841
        layer = DistributedTranspose(P_x, P_y)

        # Forward Input
        x = NoneTensor()
        if P_x.active:
            x_local_shape = compute_subsizes(P_x.comm.dims,
                                             P_x.comm.Get_coords(P_x.rank),
                                             x_global_shape)
            x = torch.Tensor(np.random.randn(*x_local_shape))
        x.requires_grad = True

        layer(x)


@pytest.mark.parametrize("comm_split_fixture", [4], indirect=["comm_split_fixture"])
def test_excepts_mismatched_nondivisible_tensor(barrier_fence_fixture,
                                                comm_split_fixture):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # A tensor with size 1 in a dimension cannot be partitioned in that
    # dimension.  (See last dimension of output and tensor.)
    in_dims = (1, 4, 1, 1)
    out_dims = (1, 1, 1, 2)
    x_global_shape = np.array([1, 16, 5, 1])

    in_size = np.prod(in_dims)
    out_size = np.prod(out_dims)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(np.arange(0, in_size))
    P_x = P_x_base.create_cartesian_topology_partition(in_dims)

    P_y_base = P_world.create_partition_inclusive(np.arange(P_world.size-out_size, P_world.size))
    P_y = P_y_base.create_cartesian_topology_partition(out_dims)

    with pytest.raises(ValueError) as e_info:  # noqa: F841
        layer = DistributedTranspose(P_x, P_y)

        # Forward Input
        x = NoneTensor()
        if P_x.active:
            x_local_shape = compute_subsizes(P_x.comm.dims,
                                             P_x.comm.Get_coords(P_x.rank),
                                             x_global_shape)
            x = torch.Tensor(np.random.randn(*x_local_shape))
        x.requires_grad = True

        layer(x)
