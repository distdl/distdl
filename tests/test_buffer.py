import pytest
from adjoint_test import check_adjoint_test_tight


@pytest.mark.parametrize("comm_split_fixture", [1], indirect=["comm_split_fixture"])
def test_buffer_expansion(barrier_fence_fixture,
                          comm_split_fixture):

    import numpy as np

    from distdl.backends.mpi.buffer import MPIExpandableBuffer

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return

    # Clean buffer should have zero size but a defined dtype
    buff = MPIExpandableBuffer(np.float32)
    assert len(buff.raw_buffer) == 0
    assert buff.raw_buffer.dtype == np.float32

    # Expanding the buffer should increase its size
    buff.expand(5)
    assert len(buff.raw_buffer) == 5
    assert buff.raw_buffer.dtype == np.float32

    # Creating a view larger than the current size should expand the buffer
    shape = (2, 3, 4)
    buff.allocate_view(shape)
    assert len(buff.raw_buffer) == 24

    # Requesting that view should create a numpy view with that shape
    # but should not change the array
    buffer_id_before = id(buff.raw_buffer)
    view = buff.get_view(shape)
    buffer_id_after = id(buff.raw_buffer)
    assert view.shape == shape
    assert buffer_id_before == buffer_id_after

    # Creating a new view of the same size should not cause reallocation
    shape = (4, 3, 2)
    buffer_id_before = id(buff.raw_buffer)
    view = buff.get_view(shape)
    buffer_id_after = id(buff.raw_buffer)
    assert view.shape == shape
    assert buffer_id_before == buffer_id_after

    # Creating a new view of smaller size should not cause reallocation
    shape = (2, 3, 2)
    buffer_id_before = id(buff.raw_buffer)
    view = buff.get_view(shape)
    buffer_id_after = id(buff.raw_buffer)
    assert view.shape == shape
    assert buffer_id_before == buffer_id_after

    # Creating a new view of larger size should cause reallocation
    shape = (4, 4, 4)
    buffer_id_before = id(buff.raw_buffer)
    view = buff.get_view(shape)
    buffer_id_after = id(buff.raw_buffer)
    assert view.shape == shape
    assert buffer_id_before != buffer_id_after

    # Filling that view, then creating a larger one should preserve the values
    # in the view, but the buffer should be a different buffer
    shape = (4, 4, 4)
    view = buff.get_view(shape)
    buffer_id_before = id(buff.raw_buffer)
    view[:] = 15.0
    old_data = view.copy()

    new_shape = (8, 8, 8)
    buff.allocate_view(new_shape)
    buffer_id_after = id(buff.raw_buffer)
    view = buff.get_view(shape)

    assert np.all(old_data == view)
    assert buffer_id_before != buffer_id_after


@pytest.mark.parametrize("comm_split_fixture", [1], indirect=["comm_split_fixture"])
def test_buffer_management(barrier_fence_fixture,
                           comm_split_fixture):

    import numpy as np

    from distdl.backends.mpi.buffer import MPIBufferManager

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return

    # A new buffer manager should be empty
    buffer_manager = MPIBufferManager()
    assert len(buffer_manager.buffers) == 0

    # Requesting new buffers should create them
    buffers = buffer_manager.request_buffers(6, np.float32)
    assert len(buffer_manager.buffers) == 6

    # Requesting a subset of buffers should get the first n of them without
    # changing the size
    buffers = buffer_manager.request_buffers(3, np.float32)
    assert len(buffer_manager.buffers) == 6
    for i in range(3):
        assert buffers[i] is buffer_manager.buffers[i]

    # Requesting new buffers of a different dtype should create them and
    # they should be different from the first group
    buffers = buffer_manager.request_buffers(6, np.int32)
    assert len(buffer_manager.buffers) == 12
    for i in range(6):
        assert buffers[i] is not buffer_manager.buffers[i]


@pytest.mark.parametrize("comm_split_fixture", [4], indirect=["comm_split_fixture"])
def test_buffer_management_transpose_network(barrier_fence_fixture,
                                             comm_split_fixture):

    import distdl
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.backends.mpi.buffer import MPIBufferManager
    from distdl.utilities.torch import zero_volume_tensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return

    buffer_manager = MPIBufferManager()
    P_world = MPIPartition(base_comm)

    # Create the partitions

    P_1_base = P_world.create_partition_inclusive([0])
    P_1 = P_1_base.create_cartesian_topology_partition([1, 1])

    P_3_base = P_world.create_partition_inclusive([1, 2, 3])
    P_3 = P_3_base.create_cartesian_topology_partition([1, 3])

    P_4_base = P_world.create_partition_inclusive([0, 1, 2, 3])
    P_4 = P_4_base.create_cartesian_topology_partition([1, 4])

    tr1 = distdl.nn.DistributedTranspose(P_1, P_4, buffer_manager=buffer_manager)
    tr2 = distdl.nn.DistributedTranspose(P_4, P_4, buffer_manager=buffer_manager)
    tr3 = distdl.nn.DistributedTranspose(P_4, P_3, buffer_manager=buffer_manager)
    tr4 = distdl.nn.DistributedTranspose(P_3, P_1, buffer_manager=buffer_manager)

    x = zero_volume_tensor(1)
    if P_1.active:
        x = torch.randn(1, 10)
    x.requires_grad = True

    # [0   1   2   3   4   5   6   7   8   9] to
    # [0   1   2] [3   4   5] [6   7] [8   9]
    x2 = tr1(x)
    n_buffers_by_rank = (3, 1, 1, 1)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    # [0   1   2] [3   4   5] [6   7] [8   9] to
    # [0   1   2] [3   4   5] [6   7] [8   9]
    x3 = tr2(x2)
    n_buffers_by_rank = (3, 1, 1, 1)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    #    [0   1   2] [3   4   5] [6   7] [8   9] to
    # [] [0   1   2   3] [4   5   6] [7   8   9]
    x4 = tr3(x3)
    n_buffers_by_rank = (3, 2, 2, 1)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    # [] [0   1   2   3] [4   5   6] [7   8   9] to
    #    [0   1   2   3   4   5   6   7   8   9]
    y = tr4(x4)
    n_buffers_by_rank = (3, 2, 2, 1)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    dy = zero_volume_tensor(1)
    if P_1.active:
        dy = torch.randn(1, 10)
    dy.requires_grad = True

    y.backward(dy)
    dx = x.grad

    # Through the backward call the buffer count do not change
    n_buffers_by_rank = (3, 2, 2, 1)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    # And adjointness is still preserved

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, x, dx, y, dy)

    P_world.deactivate()
    P_1_base.deactivate()
    P_1.deactivate()
    P_3_base.deactivate()
    P_3.deactivate()
    P_4_base.deactivate()
    P_4.deactivate()


@pytest.mark.parametrize("comm_split_fixture", [4], indirect=["comm_split_fixture"])
def test_buffer_management_conv2d_network(barrier_fence_fixture,
                                          comm_split_fixture):

    import distdl
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.backends.mpi.buffer import MPIBufferManager
    from distdl.utilities.torch import zero_volume_tensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return

    buffer_manager = MPIBufferManager()
    P_world = MPIPartition(base_comm)

    # Create the partitions

    P_1_base = P_world.create_partition_inclusive([0])
    P_1 = P_1_base.create_cartesian_topology_partition([1, 1, 1, 1])

    P_22_base = P_world.create_partition_inclusive([0, 1, 2, 3])
    P_22 = P_22_base.create_cartesian_topology_partition([1, 1, 2, 2])

    tr1 = distdl.nn.DistributedTranspose(P_1, P_22)
    c1 = distdl.nn.DistributedConv2d(P_22,
                                     in_channels=1, out_channels=5,
                                     kernel_size=[3, 3], padding=[1, 1],
                                     bias=False, buffer_manager=buffer_manager)
    c2 = distdl.nn.DistributedConv2d(P_22,
                                     in_channels=5, out_channels=10,
                                     kernel_size=[3, 3], padding=[1, 1],
                                     bias=False, buffer_manager=buffer_manager)
    c3 = distdl.nn.DistributedConv2d(P_22,
                                     in_channels=10, out_channels=20,
                                     kernel_size=[3, 3], padding=[1, 1],
                                     bias=False, buffer_manager=buffer_manager)
    tr2 = distdl.nn.DistributedTranspose(P_22, P_1)

    x = zero_volume_tensor(1)
    if P_1.active:
        x = torch.randn(1, 1, 5, 5)
    x.requires_grad = True

    # [[00   01   02   03   04]      [[00   01   02]  [[03   04]
    #  [10   11   12   13   14]       [10   11   12]   [13   14]]
    #  [20   21   22   23   24]   to  [20   21   22]] [[23   24]
    #  [30   31   32   33   34]      [[30   31   32]   [33   34]
    #  [40   41   42   43   44]]      [40   41   42]]  [43   44]]
    x2 = tr1(x)

    # [[00   01   02]  [[03   04]     [[00   01   02]  [[03   04]
    #  [10   11   12]   [13   14]]     [10   11   12]   [13   14]]
    #  [20   21   22]] [[23   24]  to  [20   21   22]] [[23   24]
    # [[30   31   32]   [33   34]     [[30   31   32]   [33   34]
    #  [40   41   42]]  [43   44]]     [40   41   42]]  [43   44]]
    x3 = c1(x2)
    n_buffers_by_rank = (4, 4, 4, 4)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    # [[00   01   02]  [[03   04]     [[00   01   02]  [[03   04]
    #  [10   11   12]   [13   14]]     [10   11   12]   [13   14]]
    #  [20   21   22]] [[23   24]  to  [20   21   22]] [[23   24]
    # [[30   31   32]   [33   34]     [[30   31   32]   [33   34]
    #  [40   41   42]]  [43   44]]     [40   41   42]]  [43   44]]
    x4 = c2(x3)
    n_buffers_by_rank = (4, 4, 4, 4)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    # [[00   01   02]  [[03   04]     [[00   01   02]  [[03   04]
    #  [10   11   12]   [13   14]]     [10   11   12]   [13   14]]
    #  [20   21   22]] [[23   24]  to  [20   21   22]] [[23   24]
    # [[30   31   32]   [33   34]     [[30   31   32]   [33   34]
    #  [40   41   42]]  [43   44]]     [40   41   42]]  [43   44]]
    x5 = c3(x4)
    n_buffers_by_rank = (4, 4, 4, 4)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    # [[00   01   02]  [[03   04]     [[00   01   02   03   04]
    #  [10   11   12]   [13   14]]     [10   11   12   13   14]
    #  [20   21   22]] [[23   24]  to  [20   21   22   23   24]
    # [[30   31   32]   [33   34]      [30   31   32   33   34]
    #  [40   41   42]]  [43   44]]     [40   41   42   43   44]]
    y = tr2(x5)

    dy = zero_volume_tensor(1)
    if P_1.active:
        dy = torch.randn(1, 20, 5, 5)
    dy.requires_grad = True

    y.backward(dy)
    dx = x.grad

    # Through the backward call the buffer count do not change
    n_buffers_by_rank = (4, 4, 4, 4)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    # And adjointness is still preserved

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, x, dx, y, dy)

    P_world.deactivate()
    P_1_base.deactivate()
    P_1.deactivate()
    P_22_base.deactivate()
    P_22.deactivate()


@pytest.mark.parametrize("comm_split_fixture", [4], indirect=["comm_split_fixture"])
def test_buffer_management_mixed_network(barrier_fence_fixture,
                                         comm_split_fixture):

    import distdl
    import torch

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.backends.mpi.buffer import MPIBufferManager
    from distdl.utilities.torch import zero_volume_tensor

    # Isolate the minimum needed ranks
    base_comm, active = comm_split_fixture
    if not active:
        return

    buffer_manager = MPIBufferManager()
    P_world = MPIPartition(base_comm)

    # Create the partitions

    P_1_base = P_world.create_partition_inclusive([0])
    P_1 = P_1_base.create_cartesian_topology_partition([1, 1, 1, 1])

    P_22_base = P_world.create_partition_inclusive([0, 1, 2, 3])
    P_22 = P_22_base.create_cartesian_topology_partition([1, 1, 2, 2])

    tr1 = distdl.nn.DistributedTranspose(P_1, P_22, buffer_manager=buffer_manager)
    c1 = distdl.nn.DistributedConv2d(P_22,
                                     in_channels=1, out_channels=5,
                                     kernel_size=[3, 3], padding=[1, 1],
                                     bias=False, buffer_manager=buffer_manager)
    c2 = distdl.nn.DistributedConv2d(P_22,
                                     in_channels=5, out_channels=10,
                                     kernel_size=[3, 3], padding=[1, 1],
                                     bias=False, buffer_manager=buffer_manager)
    c3 = distdl.nn.DistributedConv2d(P_22,
                                     in_channels=10, out_channels=20,
                                     kernel_size=[3, 3], padding=[1, 1],
                                     bias=False, buffer_manager=buffer_manager)
    tr2 = distdl.nn.DistributedTranspose(P_22, P_1, buffer_manager=buffer_manager)

    x = zero_volume_tensor(1)
    if P_1.active:
        x = torch.randn(1, 1, 5, 5)
    x.requires_grad = True

    # [[00   01   02   03   04]      [[00   01   02]  [[03   04]
    #  [10   11   12   13   14]       [10   11   12]   [13   14]]
    #  [20   21   22   23   24]   to  [20   21   22]] [[23   24]
    #  [30   31   32   33   34]      [[30   31   32]   [33   34]
    #  [40   41   42   43   44]]      [40   41   42]]  [43   44]]
    x2 = tr1(x)
    n_buffers_by_rank = (3, 1, 1, 1)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    # [[00   01   02]  [[03   04]     [[00   01   02]  [[03   04]
    #  [10   11   12]   [13   14]]     [10   11   12]   [13   14]]
    #  [20   21   22]] [[23   24]  to  [20   21   22]] [[23   24]
    # [[30   31   32]   [33   34]     [[30   31   32]   [33   34]
    #  [40   41   42]]  [43   44]]     [40   41   42]]  [43   44]]
    x3 = c1(x2)
    n_buffers_by_rank = (4, 4, 4, 4)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    # [[00   01   02]  [[03   04]     [[00   01   02]  [[03   04]
    #  [10   11   12]   [13   14]]     [10   11   12]   [13   14]]
    #  [20   21   22]] [[23   24]  to  [20   21   22]] [[23   24]
    # [[30   31   32]   [33   34]     [[30   31   32]   [33   34]
    #  [40   41   42]]  [43   44]]     [40   41   42]]  [43   44]]
    x4 = c2(x3)
    n_buffers_by_rank = (4, 4, 4, 4)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    # [[00   01   02]  [[03   04]     [[00   01   02]  [[03   04]
    #  [10   11   12]   [13   14]]     [10   11   12]   [13   14]]
    #  [20   21   22]] [[23   24]  to  [20   21   22]] [[23   24]
    # [[30   31   32]   [33   34]     [[30   31   32]   [33   34]
    #  [40   41   42]]  [43   44]]     [40   41   42]]  [43   44]]
    x5 = c3(x4)
    n_buffers_by_rank = (4, 4, 4, 4)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    # [[00   01   02]  [[03   04]     [[00   01   02   03   04]
    #  [10   11   12]   [13   14]]     [10   11   12   13   14]
    #  [20   21   22]] [[23   24]  to  [20   21   22   23   24]
    # [[30   31   32]   [33   34]      [30   31   32   33   34]
    #  [40   41   42]]  [43   44]]     [40   41   42   43   44]]
    y = tr2(x5)
    n_buffers_by_rank = (4, 4, 4, 4)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    dy = zero_volume_tensor(1)
    if P_1.active:
        dy = torch.randn(1, 20, 5, 5)
    dy.requires_grad = True

    y.backward(dy)
    dx = x.grad

    # Through the backward call the buffer count do not change
    n_buffers_by_rank = (4, 4, 4, 4)
    assert len(buffer_manager.buffers) == n_buffers_by_rank[P_world.rank]

    # And adjointness is still preserved

    x = x.detach()
    dx = dx.detach()
    dy = dy.detach()
    y = y.detach()

    check_adjoint_test_tight(P_world, x, dx, y, dy)

    P_world.deactivate()
    P_1_base.deactivate()
    P_1.deactivate()
    P_22_base.deactivate()
    P_22.deactivate()
