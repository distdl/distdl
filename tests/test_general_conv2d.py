import numpy as np
import pytest
from adjoint_test import check_adjoint_test_tight

adjoint_parametrizations = []

# Main functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 6), [1, 1, 2, 3],  # P_x_ranks, P_x_topo
        np.arange(4, 16), [1, 2, 2, 3],  # P_y_ranks, P_y_topo
        np.arange(4, 16), [2, 1, 2, 3],  # P_w_ranks, P_w_topo
        [1, 5, 10, 10],  # global_tensor_size
        16,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-co2_ci1",
        marks=[pytest.mark.mpi(min_size=16)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(4, 16), [1, 2, 2, 3],  # P_x_ranks, P_x_topo
        np.arange(0, 6), [1, 1, 2, 3],  # P_y_ranks, P_y_topo
        np.arange(4, 16), [1, 2, 2, 3],  # P_w_ranks, P_w_topo
        [1, 5, 10, 10],  # global_tensor_size
        16,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-co1_ci2",
        marks=[pytest.mark.mpi(min_size=16)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        np.arange(4, 12), [1, 2, 2, 2],  # P_x_ranks, P_x_topo
        np.arange(4, 12), [1, 2, 2, 2],  # P_y_ranks, P_y_topo
        np.arange(0, 16), [2, 2, 2, 2],  # P_w_ranks, P_w_topo
        [1, 5, 10, 10],  # global_tensor_size
        16,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-co2_ci2",
        marks=[pytest.mark.mpi(min_size=16)]
        )
    )


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_topo,"
                         "P_y_ranks, P_y_topo,"
                         "P_w_ranks, P_w_topo,"
                         "global_tensor_size,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_general_conv_adjoint_input(barrier_fence_fixture,
                                    comm_split_fixture,
                                    P_x_ranks, P_x_topo,
                                    P_y_ranks, P_y_topo,
                                    P_w_ranks, P_w_topo,
                                    global_tensor_size):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.general_conv import DistributedGeneralConv2d
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

    P_w_base = P_world.create_partition_inclusive(P_w_ranks)
    P_w = P_w_base.create_cartesian_topology_partition(P_w_topo)

    global_tensor_sizes = np.asarray(global_tensor_size)

    layer = DistributedGeneralConv2d(global_tensor_sizes,
                                     P_x, P_y, P_w,
                                     in_channels=global_tensor_sizes[1],
                                     out_channels=10,
                                     kernel_size=[3, 3], bias=False)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              global_tensor_sizes)
        x = torch.Tensor(np.random.randn(*input_tensor_sizes))
    x.requires_grad = True

    y = layer(x)

    dy = NoneTensor()
    if P_y.active:
        dy = torch.Tensor(np.random.randn(*y.shape))

    y.backward(dy)
    dx = x.grad

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, x, dx, y, dy)


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_topo,"
                         "P_y_ranks, P_y_topo,"
                         "P_w_ranks, P_w_topo,"
                         "global_tensor_size,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_general_conv_adjoint_weight(barrier_fence_fixture,
                                     comm_split_fixture,
                                     P_x_ranks, P_x_topo,
                                     P_y_ranks, P_y_topo,
                                     P_w_ranks, P_w_topo,
                                     global_tensor_size):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.general_conv import DistributedGeneralConv2d
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

    P_w_base = P_world.create_partition_inclusive(P_w_ranks)
    P_w = P_w_base.create_cartesian_topology_partition(P_w_topo)

    global_tensor_sizes = np.asarray(global_tensor_size)

    layer = DistributedGeneralConv2d(global_tensor_sizes,
                                     P_x, P_y, P_w,
                                     in_channels=global_tensor_sizes[1],
                                     out_channels=10,
                                     kernel_size=[3, 3], bias=False)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              global_tensor_sizes)
        x = torch.Tensor(np.random.randn(*input_tensor_sizes))
    x.requires_grad = True

    y = layer(x)

    dy = NoneTensor()
    if P_y.active:
        dy = torch.Tensor(np.random.randn(*y.shape))

    y.backward(dy)

    W = NoneTensor()
    dW = NoneTensor()
    if P_w.active:
        W = layer._weight.detach()
        dW = layer._weight.grad.detach()

    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, W, dW, y, dy)


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_topo,"
                         "P_y_ranks, P_y_topo,"
                         "P_w_ranks, P_w_topo,"
                         "global_tensor_size,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_general_conv_adjoint_bias(barrier_fence_fixture,
                                   comm_split_fixture,
                                   P_x_ranks, P_x_topo,
                                   P_y_ranks, P_y_topo,
                                   P_w_ranks, P_w_topo,
                                   global_tensor_size):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.general_conv import DistributedGeneralConv2d
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

    P_w_base = P_world.create_partition_inclusive(P_w_ranks)
    P_w = P_w_base.create_cartesian_topology_partition(P_w_topo)

    global_tensor_sizes = np.asarray(global_tensor_size)

    layer = DistributedGeneralConv2d(global_tensor_sizes,
                                     P_x, P_y, P_w,
                                     in_channels=global_tensor_sizes[1],
                                     out_channels=10,
                                     kernel_size=[3, 3], bias=True)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              global_tensor_sizes)
        x = torch.zeros(*input_tensor_sizes)
    x.requires_grad = True

    y = layer(x)

    dy = NoneTensor()
    if P_y.active:
        dy = torch.Tensor(np.random.randn(*y.shape))

    y.backward(dy)

    b = NoneTensor()
    db = NoneTensor()
    if P_w.active and layer.stores_bias:
        b = layer._bias.detach()
        db = layer._bias.grad.detach()

    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, b, db, y, dy)
