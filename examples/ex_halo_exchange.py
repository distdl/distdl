import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.halo_mixin import HaloMixin
from distdl.nn.padnd import PadNd
from distdl.utilities.debug import print_sequential
from distdl.utilities.misc import DummyContext
from distdl.utilities.slicing import compute_subsizes


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

P_x_base = P_world.create_partition_inclusive(use_ranks)
P_x = P_x_base.create_cartesian_topology_partition(dims)
rank = P_x.rank
cart_comm = P_x.comm

global_tensor_sizes = np.array([1, 1, 10, 12])

if P_x.active:
    mockup_conv_layer = MockupConvLayer()
    kernel_sizes = [1, 1, 3, 3]
    strides = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1, 1, 1]

    exchange_info = mockup_conv_layer._compute_exchange_info(global_tensor_sizes,
                                                             kernel_sizes,
                                                             strides,
                                                             pads,
                                                             dilations,
                                                             P_x.active,
                                                             P_x.dims,
                                                             P_x.coords)
    halo_sizes = exchange_info[0]
    recv_buffer_sizes = exchange_info[1]
    send_buffer_sizes = exchange_info[2]

    in_subsizes = compute_subsizes(P_x.comm.dims,
                                   P_x.comm.Get_coords(P_x.rank),
                                   global_tensor_sizes)

    value = (1 + rank) * (10 ** rank)
    a = np.full(shape=in_subsizes, fill_value=value, dtype=float)

    forward_input_padnd_layer = PadNd(halo_sizes.astype(int), value=0, partition=P_x)
    adjoint_input_padnd_layer = PadNd(halo_sizes.astype(int), value=value, partition=P_x)
    t = torch.tensor(a, requires_grad=True)
    t_forward_input = forward_input_padnd_layer.forward(t)
    t_adjoint_input = adjoint_input_padnd_layer.forward(t)

    halo_layer = HaloExchange(P_x, halo_sizes, recv_buffer_sizes, send_buffer_sizes)

    print_sequential(cart_comm, f'rank = {rank}, t_forward_input =\n{t_forward_input.int()}')

    ctx = DummyContext()
    t_forward_exchanged = halo_layer(t_forward_input)

    print_sequential(cart_comm, f'rank = {rank}, t_forward_exchanged =\n{t_forward_input.int()}')

    print_sequential(cart_comm, f'rank = {rank}, t_adjoint_input =\n{t_adjoint_input.int()}')

    t_forward_exchanged.backward(t_adjoint_input)

    print_sequential(cart_comm, f'rank = {rank}, t_adjoint_exchanged =\n{t_adjoint_input.int()}')
