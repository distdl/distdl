import numpy as np
import torch
from mpi4py import MPI

from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.halo_exchange import HaloExchangeFunction
from distdl.nn.halo_mixin import HaloMixin
from distdl.nn.padnd import PadNd
from distdl.utilities.debug import print_sequential
from distdl.utilities.misc import Bunch


class TestConvLayer(HaloMixin):

    # These mappings come from basic knowledge of convolutions
    def _compute_min_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take stride and dilation into account
        # padding might also not be correct in these cases...
        kernel_offsets = (kernel_sizes - 1) / 2

        # for even sized kernels, always shortchange the left side
        kernel_offsets[kernel_sizes % 2 == 0] -= 1

        bases = idx + kernel_offsets - pads
        return bases - kernel_offsets

    def _compute_max_input_range(self,
                                 idx,
                                 kernel_sizes,
                                 strides,
                                 pads,
                                 dilations):

        # incorrect, does not take stride and dilation into account
        # padding might also not be correct in these cases...
        kernel_offsets = (kernel_sizes - 1) / 2

        bases = idx + kernel_offsets - pads
        return bases + kernel_offsets


torch.set_printoptions(linewidth=200)

comm = MPI.COMM_WORLD
dims = [1, 1, 2, 2]
cart_comm = comm.Create_cart(dims=dims)
rank = cart_comm.Get_rank()

test_conv_layer = TestConvLayer()
x_in_sizes = [1, 1, 5, 6]
kernel_sizes = [1, 1, 3, 3]
strides = [1, 1, 1, 1]
pads = [0, 0, 0, 0]
dilations = [1, 1, 1, 1]

value = (1 + rank) * (10 ** rank)
a = np.full(shape=x_in_sizes, fill_value=value, dtype=float)

pad_width = [(0, 0), (0, 0), (1, 1), (1, 1)]
padnd_layer = PadNd(pad_width, value=0)
t_forward_input = torch.tensor(a, requires_grad=True)
t_forward_input = padnd_layer.forward(t_forward_input)
t_adjoint_input = t_forward_input.clone()

halo_sizes, recv_buffer_sizes, send_buffer_sizes, needed_ranges = \
    test_conv_layer._compute_exchange_info(t_forward_input.shape,
                                           kernel_sizes,
                                           strides,
                                           pads,
                                           dilations,
                                           cart_comm)

halo_layer = HaloExchange(x_in_sizes, halo_sizes, recv_buffer_sizes, send_buffer_sizes, cart_comm)

print_sequential(cart_comm, f'rank = {rank}, t_forward_input =\n{t_forward_input.int()}')

ctx = Bunch()
t_forward_exchanged = HaloExchangeFunction.forward(ctx,
                                                   t_forward_input,
                                                   halo_layer.slices,
                                                   halo_layer.buffers,
                                                   halo_layer.neighbor_ranks,
                                                   halo_layer.cart_comm)

print_sequential(cart_comm, f'rank = {rank}, t_forward_exchanged =\n{t_forward_exchanged.int()}')

print_sequential(cart_comm, f'rank = {rank}, t_adjoint_input =\n{t_adjoint_input.int()}')

t_adjoint_exchanged = HaloExchangeFunction.backward(ctx, t_adjoint_input)[0]

print_sequential(cart_comm, f'rank = {rank}, t_adjoint_exchanged =\n{t_adjoint_exchanged.int()}')
