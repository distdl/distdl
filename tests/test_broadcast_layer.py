import numpy as np
import pytest
from adjoint_test import check_adjoint_test_tight

parametrizations = []

# Main functionality
overlap_3D = \
    pytest.param(np.arange(4, 8), [2, 2, 1],  # P_x_ranks, P_x_topo
                 np.arange(0, 12), [2, 2, 3],  # P_y_ranks, P_y_topo
                 [1, 7, 5],  # global_tensor_size
                 False,  # transpose_src
                 12,  # passed to comm_split_fixture, required MPI ranks
                 id="distributed-overlap-3D",
                 marks=[pytest.mark.mpi(min_size=12)])
parametrizations.append(overlap_3D)

disjoint_3D = \
    pytest.param(np.arange(0, 4), [2, 2, 1],  # P_x_ranks, P_x_topo
                 np.arange(4, 16), [2, 2, 3],  # P_y_ranks, P_y_topo
                 [1, 7, 5],  # global_tensor_size
                 False,  # transpose_src
                 16,  # passed to comm_split_fixture, required MPI ranks
                 id="distributed-disjoint-3D",
                 marks=[pytest.mark.mpi(min_size=16)])
parametrizations.append(disjoint_3D)

disjoint_with_inactive_3D = \
    pytest.param(np.arange(0, 4), [2, 2, 1],  # P_x_ranks, P_x_topo
                 np.arange(5, 17), [2, 2, 3],  # P_y_ranks, P_y_topo
                 [1, 7, 5],  # global_tensor_size
                 False,  # transpose_src
                 17,  # passed to comm_split_fixture, required MPI ranks
                 id="distributed-disjoint-inactive-3D",
                 marks=[pytest.mark.mpi(min_size=17)])
parametrizations.append(disjoint_with_inactive_3D)

# Sequential functionality
sequential_identity = \
    pytest.param(np.arange(0, 1), [1],  # P_x_ranks, P_x_topo
                 np.arange(0, 1), [1],  # P_y_ranks, P_y_topo
                 [1, 7, 5],  # global_tensor_size
                 False,  # transpose_src
                 1,  # passed to comm_split_fixture, required MPI ranks
                 id="sequential-identity",
                 marks=[pytest.mark.mpi(min_size=1)])
parametrizations.append(sequential_identity)

# Main functionality, single source
ss_overlap_3D = \
    pytest.param(np.arange(2, 3), [1],  # P_x_ranks, P_x_topo
                 np.arange(0, 3), [1, 1, 3],  # P_y_ranks, P_y_topo
                 [1, 7, 5],  # global_tensor_size
                 False,  # transpose_src
                 3,  # passed to comm_split_fixture, required MPI ranks
                 id="distributed-overlap-3D-single_source",
                 marks=[pytest.mark.mpi(min_size=3)])
parametrizations.append(ss_overlap_3D)

ss_disjoint_3D = \
    pytest.param(np.arange(3, 4), [1],  # P_x_ranks, P_x_topo
                 np.arange(0, 3), [1, 1, 3],  # P_y_ranks, P_y_topo
                 [1, 7, 5],  # global_tensor_size
                 False,  # transpose_src
                 4,  # passed to comm_split_fixture, required MPI ranks
                 id="distributed-disjoint-3D-single_source",
                 marks=[pytest.mark.mpi(min_size=4)])
parametrizations.append(ss_disjoint_3D)

ss_disjoint_with_inactive_3D = \
    pytest.param(np.arange(4, 5), [1],  # P_x_ranks, P_x_topo
                 np.arange(0, 3), [1, 1, 3],  # P_y_ranks, P_y_topo
                 [1, 7, 5],  # global_tensor_size
                 False,  # transpose_src
                 5,  # passed to comm_split_fixture, required MPI ranks
                 id="distributed-disjoint-inactive-3D-single_source",
                 marks=[pytest.mark.mpi(min_size=5)])
parametrizations.append(ss_disjoint_with_inactive_3D)

# Main functionality, transposed
overlap_3D = \
    pytest.param(np.arange(4, 8), [2, 2, 1],  # P_x_ranks, P_x_topo
                 np.arange(0, 12), [3, 2, 2],  # P_y_ranks, P_y_topo
                 [1, 7, 5],  # global_tensor_size
                 True,  # transpose_src
                 12,  # passed to comm_split_fixture, required MPI ranks
                 id="distributed-overlap-3D",
                 marks=[pytest.mark.mpi(min_size=12)])
parametrizations.append(overlap_3D)

disjoint_3D = \
    pytest.param(np.arange(0, 4), [2, 2, 1],  # P_x_ranks, P_x_topo
                 np.arange(4, 16), [3, 2, 2],  # P_y_ranks, P_y_topo
                 [1, 7, 5],  # global_tensor_size
                 True,  # transpose_src
                 16,  # passed to comm_split_fixture, required MPI ranks
                 id="distributed-disjoint-3D",
                 marks=[pytest.mark.mpi(min_size=16)])
parametrizations.append(disjoint_3D)

disjoint_with_inactive_3D = \
    pytest.param(np.arange(0, 4), [2, 2, 1],  # P_x_ranks, P_x_topo
                 np.arange(5, 17), [3, 2, 2],  # P_y_ranks, P_y_topo
                 [1, 7, 5],  # global_tensor_size
                 True,  # transpose_src
                 17,  # passed to comm_split_fixture, required MPI ranks
                 id="distributed-disjoint-inactive-3D",
                 marks=[pytest.mark.mpi(min_size=17)])
parametrizations.append(disjoint_with_inactive_3D)


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_topo,"
                         "P_y_ranks, P_y_topo,"
                         "global_tensor_size,"
                         "transpose_src,"
                         "comm_split_fixture",
                         parametrizations,
                         indirect=["comm_split_fixture"])
def test_broadcast_adjoint(barrier_fence_fixture,
                           comm_split_fixture,
                           P_x_ranks, P_x_topo,
                           P_y_ranks, P_y_topo,
                           global_tensor_size,
                           transpose_src):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.broadcast import Broadcast
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

    # TODO #93: Change this to create a subtensor so we test when local tensors
    # have different sizes.  Then, the output size will also be different, which
    # we will have to get from `y` itself.
    tensor_sizes = np.asarray(global_tensor_size)

    layer = Broadcast(P_x, P_y, transpose_src=transpose_src)

    x = NoneTensor()
    if P_x.active:
        x = torch.Tensor(np.random.randn(*tensor_sizes))
    x.requires_grad = True

    dy = NoneTensor()
    if P_y.active:
        # Adjoint Input
        dy = torch.Tensor(np.random.randn(*tensor_sizes))

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
