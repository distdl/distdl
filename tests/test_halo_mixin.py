import numpy as np
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.halo_mixin import HaloMixin


class MockupMaxPoolLayer(HaloMixin):

    # These mappings come from the PyTorch documentation

    def _compute_min_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 stride,
                                 pads,
                                 dilations):

        # incorrect, does not take dilation and padding into account
        return stride * idx + 0

    def _compute_max_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 stride,
                                 pads,
                                 dilations):

        # incorrect, does not take dilation and padding into account
        return stride * idx + kernel_sizes - 1


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

    x_in_sizes = np.array([1, 1, 10])
    kernel_sizes = np.array([2])
    stride = np.array([2])
    pads = np.array([0])
    dilations = np.array([1])

    halo_sizes, recv_buffer_sizes, send_buffer_sizes, needed_ranges = \
        layer._compute_exchange_info(x_in_sizes,
                                     kernel_sizes,
                                     stride,
                                     pads,
                                     dilations,
                                     P_cart.active,
                                     P_cart.dims,
                                     P_cart.coords)

    if P_cart.active:
        if rank == 0:
            expected_halo_sizes = np.array([[0, 0], [0, 0], [0, 1]])
            expected_recv_buffer_sizes = np.array([[0, 0], [0, 0], [0, 1]])
            expected_send_buffer_sizes = np.array([[0, 0], [0, 0], [0, 0]])
            expected_needed_ranges = np.array([[0, 1], [0, 1], [0, 4]])

            assert(np.array_equal(halo_sizes, expected_halo_sizes))
            assert(np.array_equal(recv_buffer_sizes, expected_recv_buffer_sizes))
            assert(np.array_equal(send_buffer_sizes, expected_send_buffer_sizes))
            assert(np.array_equal(needed_ranges, expected_needed_ranges))

        elif rank == 1:
            expected_halo_sizes = np.array([[0, 0], [0, 0], [0, 0]])
            expected_recv_buffer_sizes = np.array([[0, 0], [0, 0], [0, 0]])
            expected_send_buffer_sizes = np.array([[0, 0], [0, 0], [1, 0]])
            expected_needed_ranges = np.array([[0, 1], [0, 1], [1, 3]])

            assert(np.array_equal(halo_sizes, expected_halo_sizes))
            assert(np.array_equal(recv_buffer_sizes, expected_recv_buffer_sizes))
            assert(np.array_equal(send_buffer_sizes, expected_send_buffer_sizes))
            assert(np.array_equal(needed_ranges, expected_needed_ranges))

        elif rank == 2:
            expected_halo_sizes = np.array([[0, 0], [0, 0], [0, 0]])
            expected_recv_buffer_sizes = np.array([[0, 0], [0, 0], [0, 0]])
            expected_send_buffer_sizes = np.array([[0, 0], [0, 0], [0, 0]])
            expected_needed_ranges = np.array([[0, 1], [0, 1], [0, 2]])

            assert(np.array_equal(halo_sizes, expected_halo_sizes))
            assert(np.array_equal(recv_buffer_sizes, expected_recv_buffer_sizes))
            assert(np.array_equal(send_buffer_sizes, expected_send_buffer_sizes))
            assert(np.array_equal(needed_ranges, expected_needed_ranges))

        elif rank == 3:
            expected_halo_sizes = np.array([[0, 0], [0, 0], [0, 0]])
            expected_recv_buffer_sizes = np.array([[0, 0], [0, 0], [0, 0]])
            expected_send_buffer_sizes = np.array([[0, 0], [0, 0], [0, 0]])
            expected_needed_ranges = np.array([[0, 1], [0, 1], [0, 2]])

            assert(np.array_equal(halo_sizes, expected_halo_sizes))
            assert(np.array_equal(recv_buffer_sizes, expected_recv_buffer_sizes))
            assert(np.array_equal(send_buffer_sizes, expected_send_buffer_sizes))
            assert(np.array_equal(needed_ranges, expected_needed_ranges))

    # Inactive ranks should get null results
    else:
        assert(halo_sizes is None)
        assert(recv_buffer_sizes is None)
        assert(send_buffer_sizes is None)
        assert(needed_ranges is None)
