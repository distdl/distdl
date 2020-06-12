import numpy as np
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.halo_mixin import HaloMixin
from distdl.utilities.debug import print_sequential


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


P_world = MPIPartition(MPI.COMM_WORLD)
ranks = np.arange(P_world.size)

dims = [1, 1, 4]
P_size = np.prod(dims)
use_ranks = ranks[:P_size]

P = P_world.create_subpartition(use_ranks)
P_cart = P.create_cartesian_subpartition(dims)
rank = P_cart.rank
cart_comm = P_cart.comm

layer = MockupMaxPoolLayer()

if P_cart.active:
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

    print_sequential(cart_comm, f'rank = {rank}:\nhalo_sizes =\n{halo_sizes}\n\
recv_buffer_sizes =\n{recv_buffer_sizes}\nsend_buffer_sizes =\n{send_buffer_sizes}\nneeded_ranges =\n{needed_ranges}')
