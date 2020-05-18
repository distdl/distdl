import numpy as np
from mpi4py import MPI

from distdl.nn.halo_mixin import HaloMixin
from distdl.utilities.debug import print_sequential


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

print_sequential(cart_comm, f'rank = {rank}:\nhalo_sizes =\n{halo_sizes}\n\
recv_buffer_sizes =\n{recv_buffer_sizes}\nsend_buffer_sizes =\n{send_buffer_sizes}\nneeded_ranges =\n{needed_ranges}')
