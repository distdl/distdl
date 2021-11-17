import os

import numpy as np
import pytest

# These tests aim to compare Distributed MaxPoolNd and AvgPoolNd functionality to
# PyTorch's MaxPoolNd/AvgPoolNd layers.

use_cuda = 'USE_CUDA' in os.environ

params = []

# No stride, and ideal padding for kernel size
params.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
        2,  # input_dimensions
        [1, 5, 10, 10],  # x_global_shape
        [3, 3],  # kernel_size
        [1, 1],  # padding
        [1, 1],  # stride
        [1, 1],  # dilation
        4,  # passed to comm_split_fixture, required MPI ranks
        id="no-stride-ideal-padding",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# Basic example with stride = 2 and ideal padding
params.append(
   pytest.param(
       np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
       2,  # input_dimensions
       [1, 5, 10, 10],  # x_global_shape
       [5, 5],  # kernel_size
       [2, 2],  # padding
       [2, 2],  # stride
       [1, 1],  # dilation
       4,  # passed to comm_split_fixture, required MPI ranks
       id="stride-2-ideal-padding",
       marks=[pytest.mark.mpi(min_size=4)]
       )
   )

# Odd local x shape with stride
params.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
        2,  # input_dimensions
        [1, 5, 5, 5],  # x_global_shape
        [3, 3],  # kernel_size
        [1, 1],  # padding
        [2, 2],  # stride
        [1, 1],  # dilation
        4,  # passed to comm_split_fixture, required MPI ranks
        id="odd-local-shape-with-stride",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# First kernel does not begin on local x left edge
params.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
        2,  # input_dimensions
        [1, 5, 6, 10],  # x_global_shape
        [5, 5],  # kernel_size
        [2, 2],  # padding
        [4, 4],  # stride
        [1, 1],  # dilation
        4,  # passed to comm_split_fixture, required MPI ranks
        id="kernel-needs-offset",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# Non-ideal padding
params.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
        2,  # input_dimensions
        [1, 5, 10, 8],  # x_global_shape
        [5, 5],  # kernel_size
        [1, 1],  # padding
        [1, 1],  # stride
        [1, 1],  # dilation
        4,  # passed to comm_split_fixture, required MPI ranks
        id="non-ideal-padding",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# Even kernel size
params.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
        2,  # input_dimensions
        [1, 5, 8, 8],  # x_global_shape
        [4, 4],  # kernel_size
        [1, 1],  # padding
        [1, 1],  # stride
        [1, 1],  # dilation
        4,  # passed to comm_split_fixture, required MPI ranks
        id="even-kernel-size",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# 3D input
params.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 1, 2],  # P_x_ranks, P_x_shape
        3,  # input_dimensions
        [1, 3, 5, 5, 6],  # x_global_shape
        [5, 5, 5],  # kernel_size
        [2, 2, 2],  # padding
        [2, 2, 1],  # stride
        [1, 1, 1],  # dilation
        4,  # passed to comm_split_fixture, required MPI ranks
        id="3d-input",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# 1D input
params.append(
   pytest.param(
       np.arange(0, 4), [1, 1, 4],  # P_x_ranks, P_x_shape
       1,  # input_dimensions
       [1, 3, 15],  # x_global_shape
       3,  # kernel_size
       1,  # padding
       1,  # stride
       1,  # dilation
       4,  # passed to comm_split_fixture, required MPI ranks
       id="1d-input",
       marks=[pytest.mark.mpi(min_size=4)]
       )
   )

# Dilation = 2
params.append(
   pytest.param(
       np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
       2,  # input_dimensions
       [1, 3, 10, 10],  # x_global_shape
       [5, 5],  # kernel_size
       [1, 1],  # padding
       [1, 1],  # stride
       [2, 2],  # dilation
       4,  # passed to comm_split_fixture, required MPI ranks
       id="dilation-2",
       marks=[pytest.mark.mpi(min_size=4)]
       )
   )

# Lots of partitions
params.append(
   pytest.param(
       np.arange(0, 16), [1, 1, 4, 4],  # P_x_ranks, P_x_shape
       2,  # input_dimensions
       [1, 3, 10, 10],  # x_global_shape
       [5, 5],  # kernel_size
       [2, 2],  # padding
       [2, 2],  # stride
       [1, 1],  # dilation
       16,  # passed to comm_split_fixture, required MPI ranks
       id="many-partitions-small-input",
       marks=[pytest.mark.mpi(min_size=16)]
       )
   )

# 3D input with stride, dilation, non-ideal lop-sided kernel, and large input
params.append(
    pytest.param(
        np.arange(0, 18), [1, 1, 3, 3, 2],  # P_x_ranks, P_x_shape
        3,  # input_dimensions
        [1, 5, 50, 50, 50],  # x_global_shape
        [5, 4, 3],  # kernel_size
        [1, 1, 1],  # padding
        [3, 1, 2],  # stride
        [3, 3, 1],  # dilation
        18,  # passed to comm_split_fixture, required MPI ranks
        id="hard-test",
        marks=[pytest.mark.mpi(min_size=18)]
        )
    )

# Common ResNet case
params.append(
   pytest.param(
       np.arange(0, 4), [1, 1, 1, 4],  # P_x_ranks, P_x_shape
       2,  # input_dimensions
       [10, 64, 150, 150],  # x_global_shape
       7,  # kernel_size
       3,  # padding
       2,  # stride
       1,  # dilation
       4,  # passed to comm_split_fixture, required MPI ranks
       id="common-resnet-case",
       marks=[pytest.mark.mpi(min_size=4)]
       )
   )


@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "input_dimensions,"
                         "x_global_shape,"
                         "kernel_size,"
                         "padding,"
                         "stride,"
                         "dilation,"
                         "comm_split_fixture",
                         params,
                         indirect=["comm_split_fixture"])
@pytest.mark.parametrize("layer_type", ['max', 'avg'])
def test_matches_sequential(barrier_fence_fixture,
                            comm_split_fixture,
                            P_x_ranks, P_x_shape,
                            input_dimensions,
                            x_global_shape,
                            kernel_size,
                            padding,
                            stride,
                            dilation,
                            layer_type):

    import numpy as np
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.utilities.torch import zero_volume_tensor

    device = torch.device('cuda' if use_cuda else 'cpu')

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)
    P_world._comm.Barrier()

    # Create the partitions
    P_0_base = P_world.create_partition_inclusive(np.arange(1))
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_0 = P_0_base.create_cartesian_topology_partition([1]*len(P_x_shape))
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    scatter_layer_x = DistributedTranspose(P_0, P_x).to(device)
    scatter_layer_y = DistributedTranspose(P_0, P_x).to(device)
    gather_layer_x = DistributedTranspose(P_x, P_0).to(device)
    gather_layer_y = DistributedTranspose(P_x, P_0).to(device)

    # Create the layers
    if input_dimensions == 1:
        if layer_type == 'max':
            from torch.nn import MaxPool1d as SequentialPoolType

            from distdl.nn import DistributedMaxPool1d as DistributedPoolType
        else:
            from torch.nn import AvgPool1d as SequentialPoolType

            from distdl.nn import DistributedAvgPool1d as DistributedPoolType
    elif input_dimensions == 2:
        if layer_type == 'max':
            from torch.nn import MaxPool2d as SequentialPoolType

            from distdl.nn import DistributedMaxPool2d as DistributedPoolType
        else:
            from torch.nn import AvgPool2d as SequentialPoolType

            from distdl.nn import DistributedAvgPool2d as DistributedPoolType
    elif input_dimensions == 3:
        if layer_type == 'max':
            from torch.nn import MaxPool3d as SequentialPoolType

            from distdl.nn import DistributedMaxPool3d as DistributedPoolType
        else:
            from torch.nn import AvgPool3d as SequentialPoolType

            from distdl.nn import DistributedAvgPool3d as DistributedPoolType

    # PyTorch AvgPool doesn't support dilation, so skip the test if the combination comes up
    dilation_is_default = dilation == 1 or all(x == 1 for x in dilation)
    if layer_type == 'avg' and not dilation_is_default:
        return

    layer_kwargs = {
        'kernel_size': kernel_size,
        'padding': padding,
        'stride': stride
    }

    # Only max pool layers support dilation
    if layer_type == 'max':
        layer_kwargs['dilation'] = dilation

    dist_layer = DistributedPoolType(P_x, **layer_kwargs).to(device)
    if P_0.active:
        seq_layer = SequentialPoolType(**layer_kwargs).to(device)

    # Forward Input
    x_ref = zero_volume_tensor(device=device)
    x_ref.requires_grad = True
    dy_ref = zero_volume_tensor(device=device)

    # Construct the inputs to the forward and backward functions as well as the
    # the outputs of the sequential layer
    if P_0.active:
        x_ref = torch.randn(*x_global_shape, device=device)
        x_ref.requires_grad = True
        y_ref = seq_layer(x_ref)
        y_global_shape_calc = y_ref.shape

        dy_ref = torch.randn(*y_global_shape_calc, device=device)

        y_ref.backward(dy_ref)
        dx_ref = x_ref.grad

    # Ensure that the scatter is not part of the computation we are testing
    with torch.no_grad():
        x = scatter_layer_x(x_ref.detach())
        dy = scatter_layer_y(dy_ref.detach())

    x.requires_grad = True

    y = dist_layer(x)
    y.backward(dy)
    dx = x.grad

    # Ensure that the gather is not part of the computation we are testing
    with torch.no_grad():
        dx_comp = gather_layer_x(dx.detach())
        y_comp = gather_layer_y(y.detach())

    if P_0.active:

        # Set the absolute tolerance to ~sqrt(e_mach), or the default
        # Pytorch got their defaults from NumPy, but NumPy defaults to 64-bit
        # floats, not 32-bit floats as torch does.  Consequently, the default
        # torch atol is actually tighter than one can expect from two fp-equal
        # floating point numbers.  The NumPy default of 1e-8 is closer to
        # sqrt(e_mach) for 64-bit numbers.  So we set the 32-bit tolerance to
        # a little tighter than sqrt(1e-7), 1e-5.
        if x_ref.dtype == torch.float64:
            atol = 1e-8
        elif x_ref.dtype == torch.float32:
            atol = 1e-5
        else:
            # torch default
            atol = 1e-8

        assert torch.allclose(y_ref, y_comp, atol=atol)
        assert torch.allclose(dx_ref, dx_comp, atol=atol)

    P_world.deactivate()
    P_0_base.deactivate()
    P_0.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
