import numpy as np
import pytest

import distdl

parametrizations = []

parametrizations.append(
    pytest.param(
        np.arange(0, 2), [1, 1, 2],  # P_x_ranks, P_x_shape
        None, None,  # P_y_ranks, P_y_shape
        None, None,  # P_w_ranks, P_w_shape
        [3],  # kernel_size
        distdl.nn.DistributedConv1d,
        distdl.nn.DistributedFeatureConv1d,
        2,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-conv-feature-1d",
        marks=[pytest.mark.mpi(min_size=2)]
        )
    )

parametrizations.append(
    pytest.param(
        np.arange(0, 2), [1, 2, 1],  # P_x_ranks, P_x_shape
        np.arange(2, 4), [1, 2, 1],  # P_y_ranks, P_y_shape
        np.arange(0, 4), [2, 2, 1],  # P_w_ranks, P_w_shape
        [3],  # kernel_size
        distdl.nn.DistributedConv1d,
        distdl.nn.DistributedChannelConv1d,
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-conv-channel-1d",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 2, 2],  # P_x_ranks, P_x_shape
        np.arange(2, 6), [1, 2, 2],  # P_y_ranks, P_y_shape
        np.arange(0, 8), [2, 2, 2],  # P_w_ranks, P_w_shape
        [3],  # kernel_size
        distdl.nn.DistributedConv1d,
        distdl.nn.DistributedGeneralConv1d,
        8,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-conv-general-1d",
        marks=[pytest.mark.mpi(min_size=8)]
        )
    )

parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 2],  # P_x_ranks, P_x_shape
        None, None,  # P_y_ranks, P_y_shape
        None, None,  # P_w_ranks, P_w_shape
        [3, 3],  # kernel_size
        distdl.nn.DistributedConv2d,
        distdl.nn.DistributedFeatureConv2d,
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-conv-feature-2d",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations.append(
    pytest.param(
        np.arange(0, 2), [1, 2, 1, 1],  # P_x_ranks, P_x_shape
        np.arange(2, 4), [1, 2, 1, 1],  # P_y_ranks, P_y_shape
        np.arange(0, 4), [2, 2, 1, 1],  # P_w_ranks, P_w_shape
        [3, 3],  # kernel_size
        distdl.nn.DistributedConv2d,
        distdl.nn.DistributedChannelConv2d,
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-conv-channel-2d",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 2, 1, 2],  # P_x_ranks, P_x_shape
        np.arange(2, 6), [1, 2, 1, 2],  # P_y_ranks, P_y_shape
        np.arange(0, 8), [2, 2, 1, 2],  # P_w_ranks, P_w_shape
        [3, 3],  # kernel_size
        distdl.nn.DistributedConv2d,
        distdl.nn.DistributedGeneralConv2d,
        8,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-conv-general-2d",
        marks=[pytest.mark.mpi(min_size=8)]
        )
    )

parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 1, 2, 2, 1],  # P_x_ranks, P_x_shape
        None, None,  # P_y_ranks, P_y_shape
        None, None,  # P_w_ranks, P_w_shape
        [3, 3, 3],  # kernel_size
        distdl.nn.DistributedConv3d,
        distdl.nn.DistributedFeatureConv3d,
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-conv-feature-3d",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations.append(
    pytest.param(
        np.arange(0, 2), [1, 2, 1, 1, 1],  # P_x_ranks, P_x_shape
        np.arange(2, 4), [1, 2, 1, 1, 1],  # P_y_ranks, P_y_shape
        np.arange(0, 4), [2, 2, 1, 1, 1],  # P_w_ranks, P_w_shape
        [3, 3, 3],  # kernel_size
        distdl.nn.DistributedConv3d,
        distdl.nn.DistributedChannelConv3d,
        4,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-conv-channel-3d",
        marks=[pytest.mark.mpi(min_size=4)]
        )
    )

parametrizations.append(
    pytest.param(
        np.arange(0, 4), [1, 2, 1, 1, 2],  # P_x_ranks, P_x_shape
        np.arange(2, 6), [1, 2, 1, 1, 2],  # P_y_ranks, P_y_shape
        np.arange(0, 8), [2, 2, 1, 1, 2],  # P_w_ranks, P_w_shape
        [3, 3, 3],  # kernel_size
        distdl.nn.DistributedConv3d,
        distdl.nn.DistributedGeneralConv3d,
        8,  # passed to comm_split_fixture, required MPI ranks
        id="distributed-conv-general-3d",
        marks=[pytest.mark.mpi(min_size=8)]
        )
    )


# For example of indirect, see https://stackoverflow.com/a/28570677
@pytest.mark.parametrize("P_x_ranks, P_x_shape,"
                         "P_y_ranks, P_y_shape,"
                         "P_w_ranks, P_w_shape,"
                         "kernel_size,"
                         "InputLayerType,"
                         "OutputLayerType,"
                         "comm_split_fixture",
                         parametrizations,
                         indirect=["comm_split_fixture"])
def test_conv_class_selection(barrier_fence_fixture,
                              comm_split_fixture,
                              P_x_ranks, P_x_shape,
                              P_y_ranks, P_y_shape,
                              P_w_ranks, P_w_shape,
                              kernel_size,
                              InputLayerType,
                              OutputLayerType
                              ):

    from distdl.backends.mpi.partition import MPIPartition

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)

    # Create the partitions
    P_x_base = P_world.create_partition_inclusive(P_x_ranks)
    P_x = P_x_base.create_cartesian_topology_partition(P_x_shape)

    if P_y_ranks is not None:
        P_y_base = P_world.create_partition_inclusive(P_y_ranks)
        P_y = P_y_base.create_cartesian_topology_partition(P_y_shape)
    else:
        P_y_base = None
        P_y = None

    if P_w_ranks is not None:
        P_w_base = P_world.create_partition_inclusive(P_w_ranks)
        P_w = P_w_base.create_cartesian_topology_partition(P_w_shape)
    else:
        P_w_base = None
        P_w = None

    layer = InputLayerType(P_x, P_y=P_y, P_w=P_w,
                           in_channels=3, out_channels=3,
                           kernel_size=kernel_size)

    assert type(layer) == OutputLayerType


@pytest.mark.parametrize("comm_split_fixture", [4], indirect=["comm_split_fixture"])
def test_excepts_no_match(barrier_fence_fixture,
                          comm_split_fixture):

    import numpy as np

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn import DistributedConv2d

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return
    P_world = MPIPartition(base_comm)
    P2 = P_world.create_partition_inclusive(np.arange(2))
    P4 = P_world.create_partition_inclusive(np.arange(4))

    ks = [3, 3]

    P_x = P2.create_cartesian_topology_partition([1, 2, 1, 1])
    P_y = P2.create_cartesian_topology_partition([1, 2, 1, 1])
    P_w = P4.create_cartesian_topology_partition([2, 2, 1, 1])

    with pytest.raises(ValueError) as e_info:  # noqa: F841
        layer = DistributedConv2d(P_x, P_y=P_y,  # noqa: F841
                                  in_channels=3, out_channels=3, kernel_size=ks)

    with pytest.raises(ValueError) as e_info:  # noqa: F841
        layer = DistributedConv2d(P_x, P_w=P_w,  # noqa: F841
                                  in_channels=3, out_channels=3, kernel_size=ks)
