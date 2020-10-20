import pytest


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
