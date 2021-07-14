import pytest
import torch
import os
from adjoint_test import check_adjoint_test_tight


use_cuda = 'USE_CUDA' in os.environ

adjoint_parametrizations = []

adjoint_parametrizations.append(
    pytest.param(
        [3, 4, 5],  # x_local_shape
        torch.float32,  # dtype
        [[1, 2], [1, 2], [1, 2]],  # padding
        1,  # passed to comm_split_fixture, required MPI ranks
        id="positive_padding-float32",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        [3, 4, 5],  # x_local_shape
        torch.float64,  # dtype
        [[1, 2], [1, 2], [1, 2]],  # padding
        1,  # passed to comm_split_fixture, required MPI ranks
        id="positive_padding-float64",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        [3, 4, 5],  # x_local_shape
        torch.float32,  # dtype
        [[1, 0], [0, 2], [0, 0]],  # padding
        1,  # passed to comm_split_fixture, required MPI ranks
        id="nonnegative_padding-float32",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

adjoint_parametrizations.append(
    pytest.param(
        [3, 4, 5],  # x_local_shape
        torch.float64,  # dtype
        [[1, 0], [0, 2], [0, 0]],  # padding
        1,  # passed to comm_split_fixture, required MPI ranks
        id="nonnegative_padding-float64",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )


@pytest.mark.parametrize("x_local_shape,"
                         "dtype,"
                         "padding,"
                         "comm_split_fixture",
                         adjoint_parametrizations,
                         indirect=["comm_split_fixture"])
def test_padnd_adjoint(barrier_fence_fixture,
                       comm_split_fixture,
                       x_local_shape,
                       dtype,
                       padding):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.padnd import PadNd

    device = torch.device('cuda' if use_cuda else 'cpu')

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    x_local_shape = np.asarray(x_local_shape)
    padding = np.asarray(padding)

    padded_shape = [t + lpad + rpad for t, (lpad, rpad) in zip(x_local_shape, padding)]

    layer = PadNd(padding, value=0)
    layer = layer.to(device)

    x = torch.randn(*x_local_shape, device=device).to(dtype)
    x.requires_grad = True

    dy = torch.randn(*padded_shape, device=device).to(dtype)

    y = layer(x)
    assert y.dtype == dtype

    y.backward(dy)
    dx = x.grad
    assert dx.dtype == dtype

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, x, dx, y, dy)

    P_world.deactivate()
