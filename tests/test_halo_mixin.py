import numpy as np
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.halo_mixin import HaloMixin


class MockupMaxPoolLayer(HaloMixin):

    # These mappings come from the PyTorch documentation

    def _compute_min_input_range(self,
                                 idx,
                                 kernel_size,
                                 stride,
                                 padding,
                                 dilation):

        # incorrect, does not take dilation and padding into account
        return stride * idx + 0

    def _compute_max_input_range(self,
                                 idx,
                                 kernel_size,
                                 stride,
                                 padding,
                                 dilation):

        # incorrect, does not take dilation and padding into account
        return stride * idx + kernel_size - 1


def test_mixin():

    P_world = MPIPartition(MPI.COMM_WORLD)
    ranks = np.arange(P_world.size)

    dims = [1, 1, 4]
    P_size = np.prod(dims)
    use_ranks = ranks[:P_size]

    P = P_world.create_partition_inclusive(use_ranks)
    P_cart = P.create_cartesian_topology_partition(dims)
    rank = P_cart.rank

    layer = MockupMaxPoolLayer()

    x_global_shape = np.array([1, 1, 10])
    kernel_size = np.array([2])
    stride = np.array([2])
    padding = np.array([0])
    dilation = np.array([1])

    halo_shape, recv_buffer_shape, send_buffer_shape, needed_ranges = \
        layer._compute_exchange_info(x_global_shape,
                                     kernel_size,
                                     stride,
                                     padding,
                                     dilation,
                                     P_cart.active,
                                     P_cart.dims,
                                     P_cart.coords)

    if P_cart.active:
        if rank == 0:
            expected_halo_shape = np.array([[0, 0], [0, 0], [0, 1]])
            expected_recv_buffer_shape = np.array([[0, 0], [0, 0], [0, 1]])
            expected_send_buffer_shape = np.array([[0, 0], [0, 0], [0, 0]])
            expected_needed_ranges = np.array([[0, 1], [0, 1], [0, 4]])

            assert(np.array_equal(halo_shape, expected_halo_shape))
            assert(np.array_equal(recv_buffer_shape, expected_recv_buffer_shape))
            assert(np.array_equal(send_buffer_shape, expected_send_buffer_shape))
            assert(np.array_equal(needed_ranges, expected_needed_ranges))

        elif rank == 1:
            expected_halo_shape = np.array([[0, 0], [0, 0], [0, 0]])
            expected_recv_buffer_shape = np.array([[0, 0], [0, 0], [0, 0]])
            expected_send_buffer_shape = np.array([[0, 0], [0, 0], [1, 0]])
            expected_needed_ranges = np.array([[0, 1], [0, 1], [1, 3]])

            assert(np.array_equal(halo_shape, expected_halo_shape))
            assert(np.array_equal(recv_buffer_shape, expected_recv_buffer_shape))
            assert(np.array_equal(send_buffer_shape, expected_send_buffer_shape))
            assert(np.array_equal(needed_ranges, expected_needed_ranges))

        elif rank == 2:
            expected_halo_shape = np.array([[0, 0], [0, 0], [0, 0]])
            expected_recv_buffer_shape = np.array([[0, 0], [0, 0], [0, 0]])
            expected_send_buffer_shape = np.array([[0, 0], [0, 0], [0, 0]])
            expected_needed_ranges = np.array([[0, 1], [0, 1], [0, 2]])

            assert(np.array_equal(halo_shape, expected_halo_shape))
            assert(np.array_equal(recv_buffer_shape, expected_recv_buffer_shape))
            assert(np.array_equal(send_buffer_shape, expected_send_buffer_shape))
            assert(np.array_equal(needed_ranges, expected_needed_ranges))

        elif rank == 3:
            expected_halo_shape = np.array([[0, 0], [0, 0], [0, 0]])
            expected_recv_buffer_shape = np.array([[0, 0], [0, 0], [0, 0]])
            expected_send_buffer_shape = np.array([[0, 0], [0, 0], [0, 0]])
            expected_needed_ranges = np.array([[0, 1], [0, 1], [0, 2]])

            assert(np.array_equal(halo_shape, expected_halo_shape))
            assert(np.array_equal(recv_buffer_shape, expected_recv_buffer_shape))
            assert(np.array_equal(send_buffer_shape, expected_send_buffer_shape))
            assert(np.array_equal(needed_ranges, expected_needed_ranges))

    # Inactive ranks should get null results
    else:
        assert(halo_shape is None)
        assert(recv_buffer_shape is None)
        assert(send_buffer_shape is None)
        assert(needed_ranges is None)
