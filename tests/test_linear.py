import numpy as np
import pytest
from adjoint_test import check_adjoint_test_tight

adjoint_parametrizations = []

# Main functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(4, 8), [1, 4],  # P_x_ranks, P_x_topo
        np.arange(0, 3), [1, 3],  # P_y_ranks, P_y_topo
        np.arange(0, 12), [3, 4],  # P_w_ranks, P_w_topo
        [1, 12],  # x global_tensor_size
        [1, 6],  # y global_tensor_size
        12,  # passed to comm_split_fixture, required MPI ranks
        id="distributed",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

# Main functionality
adjoint_parametrizations.append(
    pytest.param(
        np.arange(0, 1), [1, 1],  # P_x_ranks, P_x_topo
        np.arange(0, 1), [1, 1],  # P_y_ranks, P_y_topo
        np.arange(0, 1), [1, 1],  # P_w_ranks, P_w_topo
        [1, 12],  # x global_tensor_size
        [1, 6],  # y global_tensor_size
        1,  # passed to comm_split_fixture, required MPI ranks
        id="sequential",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_topo,"
                         "P_y_ranks, P_y_topo,"
                         "P_w_ranks, P_w_topo,"
                         "x_global_tensor_size,"
                         "y_global_tensor_size,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_linear_adjoint_input(barrier_fence_fixture,
                              comm_split_fixture,
                              P_x_ranks, P_x_topo,
                              P_y_ranks, P_y_topo,
                              P_w_ranks, P_w_topo,
                              x_global_tensor_size,
                              y_global_tensor_size):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.linear import Linear
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

    x_global_tensor_size = np.asarray(x_global_tensor_size)
    y_global_tensor_size = np.asarray(y_global_tensor_size)

    layer = Linear(P_x, P_y, P_w,
                   x_global_tensor_size[1],
                   y_global_tensor_size[1],
                   bias=False)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              x_global_tensor_size)
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
                         "x_global_tensor_size,"
                         "y_global_tensor_size,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_linear_adjoint_weight(barrier_fence_fixture,
                               comm_split_fixture,
                               P_x_ranks, P_x_topo,
                               P_y_ranks, P_y_topo,
                               P_w_ranks, P_w_topo,
                               x_global_tensor_size,
                               y_global_tensor_size):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.linear import Linear
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

    x_global_tensor_size = np.asarray(x_global_tensor_size)
    y_global_tensor_size = np.asarray(y_global_tensor_size)

    layer = Linear(P_x, P_y, P_w,
                   x_global_tensor_size[1],
                   y_global_tensor_size[1],
                   bias=False)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              x_global_tensor_size)
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
        W = layer.sublinear.weight.detach()
        dW = layer.sublinear.weight.grad.detach()

    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, W, dW, y, dy)


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_topo,"
                         "P_y_ranks, P_y_topo,"
                         "P_w_ranks, P_w_topo,"
                         "x_global_tensor_size,"
                         "y_global_tensor_size,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_linear_adjoint_bias(barrier_fence_fixture,
                             comm_split_fixture,
                             P_x_ranks, P_x_topo,
                             P_y_ranks, P_y_topo,
                             P_w_ranks, P_w_topo,
                             x_global_tensor_size,
                             y_global_tensor_size):
    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.linear import Linear
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

    x_global_tensor_size = np.asarray(x_global_tensor_size)
    y_global_tensor_size = np.asarray(y_global_tensor_size)

    layer = Linear(P_x, P_y, P_w,
                   x_global_tensor_size[1],
                   y_global_tensor_size[1],
                   bias=True)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              x_global_tensor_size)
        # For this test, we are only testing to see if the adjoint works
        # correctly for the bias term.  But the adjoint test only works on the
        # Jacobian of the linear layer.  The Jacobian block for b is 0 for x and
        # W, so killing x makes the forward operator equal to its Jacobian and
        # we can test to see that adjoint is computed correctly.
        x = torch.zeros(*input_tensor_sizes)
    x.requires_grad = True

    y = layer(x)

    dy = NoneTensor()
    if P_y.active:
        dy = torch.Tensor(np.random.randn(*y.shape))

    y.backward(dy)

    b = NoneTensor()
    db = NoneTensor()
    if P_w.active and P_w.coords[-1] == 0:
        b = layer.sublinear.bias.detach()
        db = layer.sublinear.bias.grad.detach()

    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, b, db, y, dy)
