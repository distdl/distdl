import numpy as np
import torch
from mpi4py import MPI

from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.halo_exchange import HaloExchangeFunction
from distdl.nn.halo_mixin import HaloMixin
from distdl.nn.padnd import PadNd
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


def test_halo_exchange_parallel_1():

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

    x = torch.tensor(np.random.randn(*x_in_sizes))
    y = torch.tensor(np.random.randn(*x_in_sizes))

    pad_width = [(0, 0), (0, 0), (1, 1), (1, 1)]
    padnd_layer = PadNd(pad_width, value=0)
    x = padnd_layer.forward(x)
    y = padnd_layer.forward(y)
    x_clone = x.clone()
    y_clone = y.clone()

    halo_sizes, recv_buffer_sizes, send_buffer_sizes, needed_ranges = \
        test_conv_layer._compute_exchange_info(x.shape,
                                               kernel_sizes,
                                               strides,
                                               pads,
                                               dilations,
                                               cart_comm)

    halo_layer = HaloExchange(x_in_sizes, halo_sizes, recv_buffer_sizes, send_buffer_sizes, cart_comm)

    ctx = Bunch()
    Ax = HaloExchangeFunction.forward(ctx,
                                      x_clone,
                                      halo_layer.slices,
                                      halo_layer.buffers,
                                      halo_layer.neighbor_ranks,
                                      halo_layer.cart_comm)
    Asy = HaloExchangeFunction.backward(ctx, y_clone)[0]

    norm_x = (torch.norm(x)**2).numpy()
    result = np.array([0.0], dtype=norm_x.dtype)
    comm.Reduce(norm_x, result, op=MPI.SUM, root=0)
    norm_x = np.sqrt(result)

    norm_y = (torch.norm(y)**2).numpy()
    result = np.array([0.0], dtype=norm_y.dtype)
    comm.Reduce(norm_y, result, op=MPI.SUM, root=0)
    norm_y = np.sqrt(result)

    norm_Ax = (torch.norm(Ax)**2).numpy()
    result = np.array([0.0], dtype=norm_Ax.dtype)
    comm.Reduce(norm_Ax, result, op=MPI.SUM, root=0)
    norm_Ax = np.sqrt(result)

    norm_Asy = (torch.norm(Asy)**2).numpy()
    result = np.array([0.0], dtype=norm_Asy.dtype)
    comm.Reduce(norm_Asy, result, op=MPI.SUM, root=0)
    norm_Asy = np.sqrt(result)

    ip1 = np.array([torch.sum(torch.mul(y, Ax))])
    result = np.array([0.0], dtype=ip1.dtype)
    comm.Reduce(ip1, result, op=MPI.SUM, root=0)
    ip1[:] = result[:]

    ip2 = np.array([torch.sum(torch.mul(Asy, x))])
    result = np.array([0.0], dtype=ip2.dtype)
    comm.Reduce(ip2, result, op=MPI.SUM, root=0)
    ip2[:] = result[:]

    # Because this is being computed in parallel, we risk that these norms
    # and inner products are not exactly equal, because the floating point
    # arithmetic is not commutative.  The only way to fix this is to collect
    # the results to a single rank to do the test.
    if(rank == 0):
        d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
        print(f"Adjoint test: {ip1/d} {ip2/d}")
        assert(np.isclose(ip1/d, ip2/d))
    else:
        # Ranks other than 0 always pass
        pass


def test_halo_exchange_parallel_2():

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world_cart = P_world.create_cartesian_subpartition([1, 1, 1, 4])
    rank = P_world_cart.rank
    cart_comm = P_world_cart.comm
    comm = cart_comm

    test_conv_layer = TestConvLayer()
    x_in_sizes = [1, 1, 5, 6]
    kernel_sizes = [1, 1, 3, 3]
    strides = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1, 1, 1]

    x = torch.tensor(np.random.randn(*x_in_sizes))
    y = torch.tensor(np.random.randn(*x_in_sizes))

    pad_width = [(0, 0), (0, 0), (1, 1), (1, 1)]
    padnd_layer = PadNd(pad_width, value=0)
    x = padnd_layer.forward(x)
    y = padnd_layer.forward(y)
    x_clone = x.clone()
    y_clone = y.clone()

    halo_sizes, recv_buffer_sizes, send_buffer_sizes, needed_ranges = \
        test_conv_layer._compute_exchange_info(x.shape,
                                               kernel_sizes,
                                               strides,
                                               pads,
                                               dilations,
                                               cart_comm)

    halo_layer = HaloExchange(x_in_sizes, halo_sizes, recv_buffer_sizes, send_buffer_sizes, P_world_cart)

    ctx = Bunch()
    Ax = HaloExchangeFunction.forward(ctx,
                                      x_clone,
                                      halo_layer.slices,
                                      halo_layer.buffers,
                                      halo_layer.neighbor_ranks,
                                      halo_layer.cart_comm)
    Asy = HaloExchangeFunction.backward(ctx, y_clone)[0]

    norm_x = (torch.norm(x)**2).numpy()
    result = np.array([0.0], dtype=norm_x.dtype)
    comm.Reduce(norm_x, result, op=MPI.SUM, root=0)
    norm_x = np.sqrt(result)

    norm_y = (torch.norm(y)**2).numpy()
    result = np.array([0.0], dtype=norm_y.dtype)
    comm.Reduce(norm_y, result, op=MPI.SUM, root=0)
    norm_y = np.sqrt(result)

    norm_Ax = (torch.norm(Ax)**2).numpy()
    result = np.array([0.0], dtype=norm_Ax.dtype)
    comm.Reduce(norm_Ax, result, op=MPI.SUM, root=0)
    norm_Ax = np.sqrt(result)

    norm_Asy = (torch.norm(Asy)**2).numpy()
    result = np.array([0.0], dtype=norm_Asy.dtype)
    comm.Reduce(norm_Asy, result, op=MPI.SUM, root=0)
    norm_Asy = np.sqrt(result)

    ip1 = np.array([torch.sum(torch.mul(y, Ax))])
    result = np.array([0.0], dtype=ip1.dtype)
    comm.Reduce(ip1, result, op=MPI.SUM, root=0)
    ip1[:] = result[:]

    ip2 = np.array([torch.sum(torch.mul(Asy, x))])
    result = np.array([0.0], dtype=ip2.dtype)
    comm.Reduce(ip2, result, op=MPI.SUM, root=0)
    ip2[:] = result[:]

    # Because this is being computed in parallel, we risk that these norms
    # and inner products are not exactly equal, because the floating point
    # arithmetic is not commutative.  The only way to fix this is to collect
    # the results to a single rank to do the test.
    if(rank == 0):
        d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
        print(f"Adjoint test: {ip1/d} {ip2/d}")
        assert(np.isclose(ip1/d, ip2/d))
    else:
        # Ranks other than 0 always pass
        pass


def test_halo_exchange_serial():

    rank = MPI.COMM_WORLD.Get_rank()

    # Isolate a single processor to use for this test.
    if rank == 0:
        color = 0
        comm = MPI.COMM_WORLD.Split(color)
    else:
        color = 1
        MPI.COMM_WORLD.Split(color)
        return

    dims = [1, 1, 1, 1]
    cart_comm = comm.Create_cart(dims=dims)

    test_conv_layer = TestConvLayer()
    x_in_sizes = [1, 1, 5, 6]
    kernel_sizes = [1, 1, 3, 3]
    strides = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1, 1, 1]

    x = torch.tensor(np.random.randn(*x_in_sizes))
    y = torch.tensor(np.random.randn(*x_in_sizes))

    pad_width = [(0, 0), (0, 0), (1, 1), (1, 1)]
    padnd_layer = PadNd(pad_width, value=0)
    x = padnd_layer.forward(x)
    y = padnd_layer.forward(y)
    x_clone = x.clone()
    y_clone = y.clone()

    halo_sizes, recv_buffer_sizes, send_buffer_sizes, needed_ranges = \
        test_conv_layer._compute_exchange_info(x.shape,
                                               kernel_sizes,
                                               strides,
                                               pads,
                                               dilations,
                                               cart_comm)

    halo_layer = HaloExchange(x_in_sizes, halo_sizes, recv_buffer_sizes, send_buffer_sizes, cart_comm)

    ctx = Bunch()
    Ax = HaloExchangeFunction.forward(ctx,
                                      x_clone,
                                      halo_layer.slices,
                                      halo_layer.buffers,
                                      halo_layer.neighbor_ranks,
                                      halo_layer.cart_comm)
    Asy = HaloExchangeFunction.backward(ctx, y_clone)[0]

    norm_x = np.sqrt((torch.norm(x) ** 2).numpy())
    norm_y = np.sqrt((torch.norm(y) ** 2).numpy())
    norm_Ax = np.sqrt((torch.norm(Ax) ** 2).numpy())
    norm_Asy = np.sqrt((torch.norm(Asy) ** 2).numpy())

    ip1 = np.array([torch.sum(torch.mul(y, Ax))])
    ip2 = np.array([torch.sum(torch.mul(Asy, x))])

    d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
    print(f"Adjoint test: {ip1/d} {ip2/d}")
    assert(np.isclose(ip1/d, ip2/d))
