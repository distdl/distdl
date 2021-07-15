import numpy as np
import pytest
from adjoint_test import check_adjoint_test_tight

adjoint_parametrizations = []

# Main functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [2, 2, 3],  # P_x_ranks, P_x_topo
        [3, 4],  # x_global_shape
        tuple(),  # axes_reduce
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-3D-0D_reduction",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [2, 2, 3],  # P_x_ranks, P_x_topo
        [3, 4],  # x_global_shape
        (0, ),  # axes_reduce
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-3D-1D_reduction",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [2, 2, 3],  # P_x_ranks, P_x_topo
        [3, 4],  # x_global_shape
        (1, ),  # axes_reduce
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-3D-1D_reduction",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [2, 2, 3],  # P_x_ranks, P_x_topo
        [3, 4],  # x_global_shape
        (2, ),  # axes_reduce
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-3D-1D_reduction",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [2, 2, 3],  # P_x_ranks, P_x_topo
        [3, 4],  # x_global_shape
        (0, 1),  # axes_reduce
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-3D-2D_reduction",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [2, 2, 3],  # P_x_ranks, P_x_topo
        [3, 4],  # x_global_shape
        (0, 2),  # axes_reduce
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-3D-2D_reduction",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [2, 2, 3],  # P_x_ranks, P_x_topo
        [3, 4],  # x_global_shape
        (1, 2),  # axes_reduce
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-3D-2D_reduction",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [2, 2, 3],  # P_x_ranks, P_x_topo
        [3, 4],  # x_global_shape
        (0, 1, 2),  # axes_reduce
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-3D-3D_reduction",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [12],  # P_x_ranks, P_x_topo
        [30, 344],  # x_global_shape
        (0, ),  # axes_reduce
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-mock_weight_reduction",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "x_global_shape,"
                         "axes_reduce,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_all_sum_reduce_adjoint(barrier_fence_fixture,
                                comm_split_fixture,
                                P_x_ranks, P_x_shape,
                                x_global_shape,
                                axes_reduce):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.all_sum_reduce import AllSumReduce
    from distdl.utilities.torch import zero_volume_tensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    # have different shape.  Then, the output size will also be different, which
    # we will have to get from `y` itself.
    x_local_shape = np.asarray(x_global_shape)

    layer = AllSumReduce(P_x, axes_reduce)

    x = zero_volume_tensor()
    if P_x.active:
        x = 10*torch.ones(*x_local_shape)
    x.requires_grad = True

    dy = zero_volume_tensor()
    if P_x.active:
        # Adjoint Input
        dy = 0.1*torch.ones(*x_local_shape)

    # y = F @ x
    y = layer(x)

    # dx = F* @ dy
    y.backward(dy)
    dx = x.grad

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    reduced_entry_value = 1
    for k in range(len(P_x_shape)):
        if k in axes_reduce:
            reduced_entry_value *= P_x_shape[k]

    assert(torch.all(y == 10*reduced_entry_value))
    assert(torch.all(dx == 0.1*reduced_entry_value))

    check_adjoint_test_tight(P_world, x, dx, y, dy)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
