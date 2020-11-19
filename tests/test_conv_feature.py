import numpy as np
import pytest

# These tests aim to compare DistributedConvNd functionality to PyTorch's ConvNd.

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
        False,  # bias
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

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)
    P_world._comm.Barrier()

    # Create the partitions
    P_root_base = P_world.create_partition_inclusive(np.arange(1))
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_root = P_root_base.create_cartesian_topology_partition([1]*len(P_x_shape))
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

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
    scatter = DistributedTranspose(P_root, P_x)
    dist_layer = dist_layer_type(P_x,
                                 in_channels=x_global_shape[1],
                                 out_channels=10,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 stride=stride,
                                 dilation=dilation,
                                 bias=bias)
    gather = DistributedTranspose(P_x, P_root)
    if P_root.active:
        seq_layer = seq_layer_type(in_channels=x_global_shape[1],
                                   out_channels=10,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   stride=stride,
                                   dilation=dilation,
                                   bias=bias)
        # set the weights of both layers to be the same
        weight = torch.rand_like(seq_layer.weight)
        seq_layer.weight.data = weight
        dist_layer.weight.data = weight
        if bias:
            bias_weight = torch.rand_like(seq_layer.bias)
            seq_layer.bias.data = bias_weight
            dist_layer.bias.data = bias_weight

    # Create the input
    if P_root.active:
        x = np.random.randn(*x_global_shape)
        dist_x = torch.from_numpy(x.copy()).to(torch.float32)
        seq_x = torch.from_numpy(x.copy()).to(torch.float32)
        dist_x.requires_grad = True
        seq_x.requires_grad = True

        if dist_x.dtype == torch.float64:
            atol = 1e-8
        elif dist_x.dtype == torch.float32:
            atol = 1e-5
        else:
            # torch default
            atol = 1e-8
    else:
        dist_x = zero_volume_tensor(requires_grad=True)

    # Check the forward pass
    dist_y = gather(dist_layer(scatter(dist_x)))
    if P_root.active:
        seq_y = seq_layer(seq_x)
        assert dist_y.shape == seq_y.shape
        assert torch.allclose(dist_y, seq_y, atol=atol)

    # Check the backward pass
    dist_y.sum().backward()
    dist_dx = dist_x.grad
    if P_root.active:
        seq_y.sum().backward()
        seq_dx = seq_x.grad
        assert dist_dx.shape == seq_dx.shape
        assert np.allclose(dist_dx, seq_dx, atol=atol)

    P_world.deactivate()
    P_x_base.deactivate()
    P_x.deactivate()
