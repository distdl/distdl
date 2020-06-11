import numpy as np
import pytest
from adjoint_test import check_adjoint_test_tight

adjoint_parametrizations = []

# Main functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [1, 1, 3, 2, 2],  # P_x_ranks, P_x_topo
        [1, 5, 10, 10, 10],  # global_tensor_size
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_topo,"
                         "global_tensor_size,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_simple_conv3d_adjoint_input(barrier_fence_fixture,
                                     comm_split_fixture,
                                     P_x_ranks, P_x_topo,
                                     global_tensor_size):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.conv import DistributedConv3d
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

    global_tensor_sizes = np.asarray(global_tensor_size)

    layer = DistributedConv3d(P_x,
                              in_channels=global_tensor_sizes[1],
                              out_channels=10,
                              kernel_size=[3, 3, 3], bias=False)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              global_tensor_sizes)
        x = torch.Tensor(np.random.randn(*input_tensor_sizes))
    x.requires_grad = True

    y = layer(x)

    dy = NoneTensor()
    if P_x.active:
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
                         "global_tensor_size,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_simple_conv3d_adjoint_weight(barrier_fence_fixture,
                                      comm_split_fixture,
                                      P_x_ranks, P_x_topo,
                                      global_tensor_size):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.conv import DistributedConv3d
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

    global_tensor_sizes = np.asarray(global_tensor_size)

    layer = DistributedConv3d(P_x,
                              in_channels=global_tensor_sizes[1],
                              out_channels=10,
                              kernel_size=[3, 3, 3], bias=False)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              global_tensor_sizes)
        x = torch.Tensor(np.random.randn(*input_tensor_sizes))
    x.requires_grad = True

    y = layer(x)

    dy = NoneTensor()
    if P_x.active:
        dy = torch.Tensor(np.random.randn(*y.shape))

    y.backward(dy)

    W = NoneTensor()
    dW = NoneTensor()
    if P_x.active:
        W = layer.weight.detach()
        dW = layer.weight.grad.detach()

    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, W, dW, y, dy)


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_topo,"
                         "global_tensor_size,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_simple_conv3d_adjoint_bias(barrier_fence_fixture,
                                    comm_split_fixture,
                                    P_x_ranks, P_x_topo,
                                    global_tensor_size):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.conv import DistributedConv3d
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

    global_tensor_sizes = np.asarray(global_tensor_size)

    layer = DistributedConv3d(P_x,
                              in_channels=global_tensor_sizes[1],
                              out_channels=10,
                              kernel_size=[3, 3, 3], bias=True)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              global_tensor_sizes)
        x = torch.zeros(*input_tensor_sizes)
    x.requires_grad = True

    y = layer(x)

    dy = NoneTensor()
    if P_x.active:
        dy = torch.Tensor(np.random.randn(*y.shape))

    y.backward(dy)

    b = NoneTensor()
    db = NoneTensor()
    if P_x.active:
        b = layer.bias.detach()
        db = layer.bias.grad.detach()

    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, b, db, y, dy)
