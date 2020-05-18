import numpy as np
from mpi4py import MPI

from distdl.nn.halo_mixin import HaloMixin


class TestMaxPoolLayer(HaloMixin):

    # These mappings come from the PyTorch documentation

    def _compute_min_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take dilation and padding into account
        return strides * idx + 0

    def _compute_max_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take dilation and padding into account
        return strides * idx + kernel_sizes - 1


def test_mixin():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    dims = [1, 1, 4]
    cart_comm = comm.Create_cart(dims=dims)

    layer = TestMaxPoolLayer()

    x_in_sizes = np.array([1, 1, 10])
    kernel_sizes = np.array([2])
    strides = np.array([2])
    pads = np.array([0])
    dilations = np.array([1])

    halo_sizes, recv_buffer_sizes, send_buffer_sizes, needed_ranges = \
        layer._compute_exchange_info(x_in_sizes,
                                     kernel_sizes,
                                     strides,
                                     pads,
                                     dilations,
                                     cart_comm)

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
