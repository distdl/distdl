import numpy as np
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.halo_mixin import HaloMixin
from distdl.utilities.debug import print_sequential


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

    print_sequential(cart_comm, f'rank = {rank}:\nhalo_shape =\n{halo_shape}\n\
recv_buffer_shape =\n{recv_buffer_shape}\nsend_buffer_shape =\n{send_buffer_shape}\nneeded_ranges =\n{needed_ranges}')
