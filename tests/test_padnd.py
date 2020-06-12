import pytest
from adjoint_test import check_adjoint_test_tight

adjoint_parametrizations = []

adjoint_parametrizations.append(
    pytest.param(
        [3, 4, 5],  # x_local_shape
        [[1, 2], [1, 2], [1, 2]],  # padding
        1,  # passed to comm_split_fixture, required MPI ranks
        id="positive_padding",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        [3, 4, 5],  # x_local_shape
        [[1, 0], [0, 2], [0, 0]],  # padding
        1,  # passed to comm_split_fixture, required MPI ranks
        id="nonnegative_padding",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )


@pytest.mark.parametrize("x_local_shape,"
                         "padding,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_padnd_adjoint(barrier_fence_fixture,
                       comm_split_fixture,
                       x_local_shape,
                       padding):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.padnd import PadNd

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    x_local_shape = np.asarray(x_local_shape)
    padding = np.asarray(padding)

    padded_shape = [t + lpad + rpad for t, (lpad, rpad) in zip(x_local_shape, padding)]

    layer = PadNd(padding, value=0)

    x = torch.tensor(np.random.randn(*x_local_shape))
    x.requires_grad = True

    dy = torch.tensor(np.random.randn(*padded_shape))

    y = layer(x)
    y.backward(dy)
    dx = x.grad

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, x, dx, y, dy)
