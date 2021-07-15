import numpy as np
import pytest
import os

# These tests aim to compare DistributedConvNd functionality to PyTorch's ConvNd.

use_cuda = 'USE_CUDA' in os.environ
assert use_cuda == True

params = []

# No stride, and ideal padding for kernel size with scalar inputs
params.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
        2,  # input_dimensions
        [1, 5, 10, 10],  # x_global_shape
        3,  # kernel_size
        1,  # padding
        1,  # stride
        1,  # dilation
        False,  # bias
        4,  # passed to comm_split_fixture, required MPI ranks
        id="no-stride-ideal-padding",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# Basic example with stride = 2 and ideal padding with scalar inputs
params.append(
   pytest.param(
       np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
       2,  # input_dimensions
       [1, 5, 10, 10],  # x_global_shape
       5,  # kernel_size
       2,  # padding
       2,  # stride
       1,  # dilation
       False,  # bias
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
        False,  # bias
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
        False,  # bias
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
        False,  # bias
        4,  # passed to comm_split_fixture, required MPI ranks
        id="non-ideal-padding",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# Even kernel size with bias
params.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
        2,  # input_dimensions
        [1, 5, 8, 8],  # x_global_shape
        [4, 4],  # kernel_size
        [1, 1],  # padding
        [1, 1],  # stride
        [1, 1],  # dilation
        True,  # bias
        4,  # passed to comm_split_fixture, required MPI ranks
        id="even-kernel-size-with-bias",
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
        False,  # bias
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
       False,  # bias
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
       False,  # bias
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
       False,  # bias
       16,  # passed to comm_split_fixture, required MPI ranks
       id="many-partitions-small-input",
       marks=[pytest.mark.mpi(min_size=16)]
       )
   )

# With bias
params.append(
   pytest.param(
       np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
       2,  # input_dimensions
       [1, 5, 10, 10],  # x_global_shape
       [3, 3],  # kernel_size
       [1, 1],  # padding
       [1, 1],  # stride
       [1, 1],  # dilation
       True,  # bias
       4,  # passed to comm_split_fixture, required MPI ranks
       id="with-bias",
       marks=[pytest.mark.mpi(min_size=4)]
       )
   )

# 3D input with bias, stride, dilation, non-ideal lop-sided kernel, and large input
params.append(
    pytest.param(
        np.arange(0, 18), [1, 1, 3, 3, 2],  # P_x_ranks, P_x_shape
        3,  # input_dimensions
        [1, 5, 50, 50, 50],  # x_global_shape
        [5, 4, 3],  # kernel_size
        [1, 1, 2],  # padding
        [3, 1, 2],  # stride
        [3, 3, 1],  # dilation
        True,  # bias
        18,  # passed to comm_split_fixture, required MPI ranks
        id="hard-test",
        marks=[pytest.mark.mpi(min_size=18)]
        )
    )

# Common conv layer for ResNet
params.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
        2,  # input_dimensions
        [1, 5, 20, 20],  # x_global_shape
        7,  # kernel_size
        3,  # padding
        2,  # stride
        1,  # dilation
        False,  # bias
        4,  # passed to comm_split_fixture, required MPI ranks
        id="common-resnet-test",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

# serial case
params.append(
    pytest.param(
        np.arange(0, 1), [1, 1, 1, 1],  # P_x_ranks, P_x_shape
        2,  # input_dimensions
        [1, 3, 16, 16],  # x_global_shape
        3,  # kernel_size
        1,  # padding
        1,  # stride
        1,  # dilation
        False,  # bias
        1,  # passed to comm_split_fixture, required MPI ranks
        id="serial-test",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )


@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "input_dimensions,"
                         "x_global_shape,"
                         "kernel_size,"
                         "padding,"
                         "stride,"
                         "dilation,"
                         "bias,"
                         "comm_split_fixture",
                         params,
                         indirect=["comm_split_fixture"])
def test_conv_versus_pytorch(barrier_fence_fixture,
                             comm_split_fixture,
                             P_x_ranks, P_x_shape,
                             input_dimensions,
                             x_global_shape,
                             kernel_size,
                             padding,
                             stride,
                             dilation,
                             bias):

    import numpy as np
    import torch
    from torch.nn import Conv1d
    from torch.nn import Conv2d
    from torch.nn import Conv3d

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.conv_feature import DistributedFeatureConv1d
    from distdl.nn.conv_feature import DistributedFeatureConv2d
    from distdl.nn.conv_feature import DistributedFeatureConv3d
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

    scatter_layer_x = DistributedTranspose(P_0, P_x)
    scatter_layer_x = scatter_layer_x.to(device)
    scatter_layer_y = DistributedTranspose(P_0, P_x)
    scatter_layer_y = scatter_layer_y.to(device)
    gather_layer_x = DistributedTranspose(P_x, P_0)
    gather_layer_x = gather_layer_x.to(device)
    gather_layer_y = DistributedTranspose(P_x, P_0)
    gather_layer_y = gather_layer_y.to(device)

    # Create the layers
    if input_dimensions == 1:
        dist_layer_type = DistributedFeatureConv1d
        seq_layer_type = Conv1d
    elif input_dimensions == 2:
        dist_layer_type = DistributedFeatureConv2d
        seq_layer_type = Conv2d
    elif input_dimensions == 3:
        dist_layer_type = DistributedFeatureConv3d
        seq_layer_type = Conv3d
    dist_layer = dist_layer_type(P_x,
                                 in_channels=x_global_shape[1],
                                 out_channels=10,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 stride=stride,
                                 dilation=dilation,
                                 bias=bias)
    dist_layer = dist_layer.to(device)
    if P_0.active:
        seq_layer = seq_layer_type(in_channels=x_global_shape[1],
                                   out_channels=10,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   stride=stride,
                                   dilation=dilation,
                                   bias=bias)
        # set the weights of both layers to be the same
        seq_layer = seq_layer.to(device)
        weight = torch.rand_like(seq_layer.weight, device=device)
        seq_layer.weight.data = weight
        dist_layer.weight.data = weight
        if bias:
            bias_weight = torch.rand_like(seq_layer.bias, device=device)
            seq_layer.bias.data = bias_weight
            dist_layer.bias.data = bias_weight

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

        # Test the result of each entry independently
        assert torch.allclose(y_ref, y_comp, atol=atol)
        assert torch.allclose(dx_ref, dx_comp, atol=atol)

    P_world.deactivate()
    P_0_base.deactivate()
    P_0.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
