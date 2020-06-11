import pytest
from adjoint_test import check_adjoint_test_tight

adjoint_parametrizations = []

adjoint_parametrizations.append(
    pytest.param(
        [4, 5, 6],  # tensor_sizes
        [[1, 2], [1, 2], [1, 2]],  # pads
        1,  # passed to comm_split_fixture, required MPI ranks
        id="positive_padding",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        [4, 5, 6],  # tensor_sizes
        [[1, 0], [0, 2], [0, 0]],  # pads
        1,  # passed to comm_split_fixture, required MPI ranks
        id="nonnegative_padding",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )


@pytest.mark.parametrize("tensor_sizes,"
                         "pads,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_unpadnd_adjoint(barrier_fence_fixture,
                         comm_split_fixture,
                         tensor_sizes,
                         pads):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.unpadnd import UnPadNd
    from distdl.utilities.torch import NoneTensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)
    P = P_world.create_partition_inclusive([0])

    tensor_sizes = np.asarray(tensor_sizes)
    pads = np.asarray(pads)

    layer = UnPadNd(pads, value=0, partition=P)

    x = NoneTensor()
    if P.active:
        x = torch.tensor(np.random.randn(*tensor_sizes))
    x.requires_grad = True

    y = layer(x)

    dy = NoneTensor()
    if P.active:
        dy = torch.tensor(np.random.randn(*y.shape))

    y.backward(dy)
    dx = x.grad

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, x, dx, y, dy)
