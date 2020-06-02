import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.halo_exchange import HaloExchangeFunction
from distdl.nn.halo_mixin import HaloMixin
from distdl.nn.padnd import PadNd
from distdl.utilities.debug import print_sequential
from distdl.utilities.misc import DummyContext


class MockupConvLayer(HaloMixin):

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

P_world = MPIPartition(MPI.COMM_WORLD)
ranks = np.arange(P_world.size)

dims = [1, 1, 2, 2]
P_size = np.prod(dims)
use_ranks = ranks[:P_size]

P = P_world.create_subpartition(use_ranks)
P_cart = P.create_cartesian_subpartition(dims)
rank = P_cart.rank
cart_comm = P_cart.comm

if P_cart.active:
    mockup_conv_layer = MockupConvLayer()
    x_in_sizes = [1, 1, 5, 6]
    kernel_sizes = [1, 1, 3, 3]
    strides = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1, 1, 1]

    halo_sizes, recv_buffer_sizes, send_buffer_sizes, needed_ranges = \
        mockup_conv_layer._compute_exchange_info(x_in_sizes,
                                                 kernel_sizes,
                                                 strides,
                                                 pads,
                                                 dilations,
                                                 P_cart.active,
                                                 P_cart.dims,
                                                 P_cart.coords)

    value = (1 + rank) * (10 ** rank)
    a = np.full(shape=x_in_sizes, fill_value=value, dtype=float)

    forward_input_padnd_layer = PadNd(halo_sizes.astype(int), value=0)
    adjoint_input_padnd_layer = PadNd(halo_sizes.astype(int), value=value)
    t = torch.tensor(a, requires_grad=True)
    t_forward_input = forward_input_padnd_layer.forward(t)
    t_adjoint_input = adjoint_input_padnd_layer.forward(t)

    halo_layer = HaloExchange(t_forward_input.shape, halo_sizes, recv_buffer_sizes, send_buffer_sizes, P_cart)

    print_sequential(cart_comm, f'rank = {rank}, t_forward_input =\n{t_forward_input.int()}')

    ctx = DummyContext()
    t_forward_exchanged = HaloExchangeFunction.forward(ctx,
                                                       t_forward_input,
                                                       halo_layer.slices,
                                                       halo_layer.buffers,
                                                       halo_layer.neighbor_ranks,
                                                       halo_layer.cartesian_partition)

    print_sequential(cart_comm, f'rank = {rank}, t_forward_exchanged =\n{t_forward_exchanged.int()}')

    print_sequential(cart_comm, f'rank = {rank}, t_adjoint_input =\n{t_adjoint_input.int()}')

    t_adjoint_exchanged = HaloExchangeFunction.backward(ctx, t_adjoint_input)[0]

    print_sequential(cart_comm, f'rank = {rank}, t_adjoint_exchanged =\n{t_adjoint_exchanged.int()}')
