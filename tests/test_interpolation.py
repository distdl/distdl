import pytest
import torch
from adjoint_test import check_adjoint_test_tight_sequential

gradcheck_parametrizations = []

# Main functionality
gradcheck_parametrizations.append(
    pytest.param(
        [3, 4, 7, 13, 17], [0, 0, 0, 0, 0], [3, 4, 7, 13, 17],  # x_global_shape, x_start, x_stop
        [3, 4, 7, 13, 17], [0, 0, 0, 0, 0], [3, 4, 7, 13, 17],  # y_global_shape, y_start, y_stop
        1,  # passed to comm_split_fixture, required MPI ranks
        id="same-full_x-full_y",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

gradcheck_parametrizations.append(
    pytest.param(
        [3, 4, 7, 13, 17], [0, 0, 0, 0, 0], [3, 4, 7, 13, 17],  # x_global_shape, x_start, x_stop
        [3, 4, 14, 26, 34], [0, 0, 0, 0, 0], [3, 4, 14, 26, 34],  # y_global_shape, y_start, y_stop
        1,  # passed to comm_split_fixture, required MPI ranks
        id="2x-full_x-full_y",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

gradcheck_parametrizations.append(
    pytest.param(
        [3, 4, 7, 13, 17], [0, 0, 0, 0, 0], [3, 4, 7, 13, 17],  # x_global_shape, x_start, x_stop
        [3, 4, 11, 20, 24], [0, 0, 3, 6, 9], [3, 4, 7, 12, 19],  # y_global_shape, y_start, y_stop
        1,  # passed to comm_split_fixture, required MPI ranks
        id="1.5x-full_x-partial_y",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

gradcheck_parametrizations.append(
    pytest.param(
        [3, 4, 7, 13, 17], [0, 0, 2, 4, 9], [3, 4, 7, 10, 12],  # x_global_shape, x_start, x_stop
        [3, 4, 11, 20, 24], [0, 0, 0, 0, 0], [3, 4, 11, 20, 24],  # y_global_shape, y_start, y_stop
        1,  # passed to comm_split_fixture, required MPI ranks
        id="1.5x-partial_x-full_y",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

gradcheck_parametrizations.append(
    pytest.param(
        [3, 4, 7, 13, 17], [0, 0, 2, 4, 9], [3, 4, 7, 10, 12],  # x_global_shape, x_start, x_stop
        [3, 4, 11, 20, 24], [0, 0, 3, 6, 9], [3, 4, 7, 12, 19],  # y_global_shape, y_start, y_stop
        1,  # passed to comm_split_fixture, required MPI ranks
        id="1.5x-partial_x-partial_y",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("x_global_shape, x_start, x_stop,"
                         "y_global_shape, y_start, y_stop,"
                         "comm_split_fixture",
                         gradcheck_parametrizations,
                         indirect=["comm_split_fixture"])
@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["constant", "nearest", "linear"])
@pytest.mark.parametrize("align_corners", [True, False])
def test_interpolation_adjoint(barrier_fence_fixture,
                               comm_split_fixture,
                               x_global_shape, x_start, x_stop,
                               y_global_shape, y_start, y_stop,
                               dimension,
                               dtype, mode, align_corners):

    import torch

    # from torch.autograd.gradcheck import gradcheck
    from distdl.nn.interpolate import Interpolate

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return

    # Test align_corners only in the linear case, otherwise ignore it.
    if mode != "linear" and align_corners:
        return

    dim_range = 2 + dimension

    # Select out the valid dimensions
    x_global_shape = x_global_shape[:dim_range]
    x_start = x_start[:dim_range]
    x_stop = x_stop[:dim_range]
    x_local_shape = tuple([b-a for a, b in zip(x_start, x_stop)])

    y_global_shape = y_global_shape[:dim_range]
    y_start = y_start[:dim_range]
    y_stop = y_stop[:dim_range]
    y_local_shape = tuple([b-a for a, b in zip(y_start, y_stop)])

    x = torch.rand(*x_local_shape, dtype=dtype)
    x.requires_grad = True

    dy = torch.randn(*y_local_shape)

    layer = Interpolate(x_start, x_stop, x_global_shape,
                        y_start, y_stop, y_global_shape,
                        mode=mode,
                        align_corners=align_corners)

    # y = F @ x
    y = layer(x)

    # dx = F* @ dy
    y.backward(dy)
    dx = x.grad

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    # Adjoint test is about 50x faster than gradcheck and this is a linear
    # operator, so adjoint test is sufficient and lets us test both larger
    # operators and both dtypes.  Gradcheck requires doubles.
    check_adjoint_test_tight_sequential(x, dx, y, dy)
    # input = [x]
    # assert gradcheck(layer, input, eps=1e-6, atol=1e-4)
