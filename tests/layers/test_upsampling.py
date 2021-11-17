import numpy as np
import pytest

match_parametrizations = []

# Main functionality
match_parametrizations.append(
    pytest.param(
        np.arange(0, 5), [1, 1, 5],  # P_x_ranks, P_x_shape
        [3, 4, 17],  # x_global_shape
        1,  # scale_factor
        [3, 4, 17],  # y_global_shape
        5,  # passed to comm_split_fixture, required MPI ranks
        id="1D-1x-full_x-full_y",
        marks=[pytest.mark.mpi(min_size=5)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 5), [1, 1, 5],  # P_x_ranks, P_x_shape
        [3, 4, 17],  # x_global_shape
        2,  # scale_factor
        [3, 4, 34],  # y_global_shape
        5,  # passed to comm_split_fixture, required MPI ranks
        id="1D-2x-full_x-full_y",
        marks=[pytest.mark.mpi(min_size=5)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 5), [1, 1, 5],  # P_x_ranks, P_x_shape
        [3, 4, 17],  # x_global_shape
        1.5,  # scale_factor
        [3, 4, 24],  # y_global_shape
        5,  # passed to comm_split_fixture, required MPI ranks
        id="1D-1.5x-full_x-partial_y",
        marks=[pytest.mark.mpi(min_size=5)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 5), [1, 1, 5],  # P_x_ranks, P_x_shape
        [3, 4, 17],  # x_global_shape
        1.5,  # scale_factor
        [3, 4, 24],  # y_global_shape
        5,  # passed to comm_split_fixture, required MPI ranks
        id="1D-1.5x-partial_x-full_y",
        marks=[pytest.mark.mpi(min_size=5)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 5), [1, 1, 5],  # P_x_ranks, P_x_shape
        [3, 4, 17],  # x_global_shape
        1.5,  # scale_factor
        [3, 4, 24],  # y_global_shape
        5,  # passed to comm_split_fixture, required MPI ranks
        id="1D-1.5x-partial_x-partial_y",
        marks=[pytest.mark.mpi(min_size=5)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [1, 1, 3, 4],  # P_x_ranks, P_x_shape
        [3, 4, 13, 17],  # x_global_shape
        1,  # scale_factor
        [3, 4, 13, 17],  # y_global_shape
        12,  # passed to comm_split_fixture, required MPI ranks
        id="2D-1x-full_x-full_y",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [1, 1, 3, 4],  # P_x_ranks, P_x_shape
        [3, 4, 13, 17],  # x_global_shape
        2,  # scale_factor
        [3, 4, 26, 34],  # y_global_shape
        12,  # passed to comm_split_fixture, required MPI ranks
        id="2D-2x-full_x-full_y",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [1, 1, 3, 4],  # P_x_ranks, P_x_shape
        [3, 4, 13, 17],  # x_global_shape
        1.5,  # scale_factor
        [3, 4, 20, 24],  # y_global_shape
        12,  # passed to comm_split_fixture, required MPI ranks
        id="2D-1.5x-full_x-partial_y",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [1, 1, 3, 4],  # P_x_ranks, P_x_shape
        [3, 4, 13, 17],  # x_global_shape
        1.5,  # scale_factor
        [3, 4, 20, 24],  # y_global_shape
        12,  # passed to comm_split_fixture, required MPI ranks
        id="2D-1.5x-partial_x-full_y",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 12), [1, 1, 3, 4],  # P_x_ranks, P_x_shape
        [3, 4, 13, 17],  # x_global_shape
        1.5,  # scale_factor
        [3, 4, 20, 24],  # y_global_shape
        12,  # passed to comm_split_fixture, required MPI ranks
        id="2D-1.5x-partial_x-partial_y",
        marks=[pytest.mark.mpi(min_size=12)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 18), [1, 1, 2, 3, 3],  # P_x_ranks, P_x_shape
        [3, 4, 7, 13, 17],  # x_global_shape
        1,  # scale_factor
        [3, 4, 7, 13, 17],  # y_global_shape
        18,  # passed to comm_split_fixture, required MPI ranks
        id="3D-1x-full_x-full_y",
        marks=[pytest.mark.mpi(min_size=18)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 18), [1, 1, 2, 3, 3],  # P_x_ranks, P_x_shape
        [3, 4, 7, 13, 17],  # x_global_shape
        2,  # scale_factor
        [3, 4, 14, 26, 34],  # y_global_shape
        18,  # passed to comm_split_fixture, required MPI ranks
        id="3D-2x-full_x-full_y",
        marks=[pytest.mark.mpi(min_size=18)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 18), [1, 1, 2, 3, 3],  # P_x_ranks, P_x_shape
        [3, 4, 7, 13, 17],  # x_global_shape
        1.5,  # scale_factor
        [3, 4, 11, 20, 24],  # y_global_shape
        18,  # scale_factor
        id="3D-1.5x-full_x-partial_y",
        marks=[pytest.mark.mpi(min_size=1)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 18), [1, 1, 2, 3, 3],  # P_x_ranks, P_x_shape
        [3, 4, 7, 13, 17],  # x_global_shape
        1.5,  # scale_factor
        [3, 4, 11, 20, 24],  # y_global_shape
        18,  # passed to comm_split_fixture, required MPI ranks
        id="3D-1.5x-partial_x-full_y",
        marks=[pytest.mark.mpi(min_size=18)]
        )
    )

match_parametrizations.append(
    pytest.param(
        np.arange(0, 18), [1, 1, 2, 3, 3],  # P_x_ranks, P_x_shape
        [3, 4, 7, 13, 17],  # x_global_shape
        1.5,  # scale_factor
        [3, 4, 11, 20, 24],  # y_global_shape
        18,  # passed to comm_split_fixture, required MPI ranks
        id="3D-1.5x-partial_x-partial_y",
        marks=[pytest.mark.mpi(min_size=18)]
        )
    )

interp_parametrizations = list()
# (mode, align_corners)
interp_parametrizations.append(pytest.param("nearest",  False))
interp_parametrizations.append(pytest.param("linear",  True))
interp_parametrizations.append(pytest.param("linear",  False))


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "x_global_shape, "
                         "scale_factor, "
                         "y_global_shape, "
                         "comm_split_fixture",
                         match_parametrizations,
                         indirect=["comm_split_fixture"])
@pytest.mark.parametrize("mode, align_corners", interp_parametrizations)
@pytest.mark.parametrize("use_size", [True, False])
def test_upsample_matches_sequential(barrier_fence_fixture,
                                     comm_split_fixture,
                                     P_x_ranks, P_x_shape,
                                     x_global_shape,
                                     scale_factor,
                                     y_global_shape,
                                     mode, align_corners,
                                     use_size):

    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.nn.upsampling import DistributedUpsample
    from distdl.utilities.torch import zero_volume_tensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return

    # Test align_corners only in the linear case, otherwise ignore it.
    if mode != "linear" and align_corners:
        return

    torch_mode_map = {3: "linear",
                      4: "bilinear",
                      5: "trilinear"}
    torch_mode = mode
    if mode == "linear":
        torch_mode = torch_mode_map[len(x_global_shape)]

    torch_align_corners = align_corners
    if mode == "nearest":
        torch_align_corners = None

    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_0_base = P_world.create_partition_inclusive([0])
    P_0 = P_0_base.create_cartesian_topology_partition([1]*len(P_x_shape))

    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    scatter_layer_x = DistributedTranspose(P_0, P_x)
    scatter_layer_y = DistributedTranspose(P_0, P_x)
    gather_layer_x = DistributedTranspose(P_x, P_0)
    gather_layer_y = DistributedTranspose(P_x, P_0)

    if use_size:
        dist_layer = DistributedUpsample(P_x, size=y_global_shape, mode=mode, align_corners=align_corners)
        if P_0.active:
            seq_layer = torch.nn.Upsample(size=y_global_shape[2:], mode=torch_mode, align_corners=torch_align_corners)
    else:
        dist_layer = DistributedUpsample(P_x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        if P_0.active:
            seq_layer = torch.nn.Upsample(scale_factor=scale_factor, mode=torch_mode, align_corners=torch_align_corners)

    # Forward Input
    x_ref = zero_volume_tensor()
    x_ref.requires_grad = True
    dy_ref = zero_volume_tensor()

    # Construct the inputs to the forward and backward functions as well as the
    # the outputs of the sequential layer
    if P_0.active:
        x_ref = torch.randn(*x_global_shape)
        x_ref.requires_grad = True
        y_ref = seq_layer(x_ref)
        y_global_shape_calc = y_ref.shape

        dy_ref = torch.randn(*y_global_shape_calc)

        y_ref.backward(dy_ref)
        dx_ref = x_ref.grad

    # Ensure that the scatter is not part of the computation we are testing
    with torch.no_grad():
        x = scatter_layer_x(x_ref.detach())
        dy = scatter_layer_y(dy_ref.detach())

    x.requires_grad = True

    # Because  there is no guarantee that any padding is needed, in this test,
    # the input x may pass directly to the Halo layer without going through
    # the padding process.  As the halo layer is in-place, that would mean a leaf-node
    # variable is modified in-place, which PyTorch does not allow.
    #
    # Thus, we have to clone it to make the input not a leaf-node.
    x_clone = x.clone()
    y = dist_layer(x_clone)
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
