import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.halo_exchange import HaloExchangeFunction
from distdl.nn.halo_mixin import HaloMixin
from distdl.nn.padnd import PadNd
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


def test_halo_exchange_parallel():

    P_world = MPIPartition(MPI.COMM_WORLD)
    ranks = np.arange(P_world.size)

    dims = [1, 1, 2, 2]
    P_size = np.prod(dims)
    use_ranks = ranks[:P_size]

    P = P_world.create_subpartition(use_ranks)
    P_cart = P.create_cartesian_subpartition(dims)

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
                                                 P_cart)

    if P_cart.active:
        x = torch.tensor(np.random.randn(*x_in_sizes))
        forward_padnd_layer = PadNd(halo_sizes.astype(int), value=0)
        x = forward_padnd_layer.forward(x)
        x_clone = x.clone()

        tensor_sizes = x_clone.shape

        y = torch.tensor(np.random.randn(*x_in_sizes))
        # Value should be random but PadNd cannot do random padding for each element. Setting
        # it to a nonzero value will be enough to guarantee correctness of the adjoint test.
        adjoint_padnd_layer = PadNd(halo_sizes.astype(int), value=1)
        y = adjoint_padnd_layer.forward(y)
        y_clone = y.clone()
    else:
        x = None
        x_clone = None

        tensor_sizes = None

        y = None
        y_clone = None

    halo_layer = HaloExchange(tensor_sizes, halo_sizes,
                              recv_buffer_sizes, send_buffer_sizes, P_cart)

    ctx = DummyContext()

    Ax = HaloExchangeFunction.forward(ctx,
                                      x_clone,
                                      halo_layer.slices,
                                      halo_layer.buffers,
                                      halo_layer.neighbor_ranks,
                                      halo_layer.cartesian_partition)

    Asy = HaloExchangeFunction.backward(ctx, y_clone)[0]

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    if P_cart.active:
        local_results[0] = (torch.norm(x)**2).numpy()
        local_results[1] = (torch.norm(y)**2).numpy()
        local_results[2] = (torch.norm(Ax)**2).numpy()
        local_results[3] = (torch.norm(Asy)**2).numpy()
        local_results[4] = np.array([torch.sum(torch.mul(Ax, y))])
        local_results[5] = np.array([torch.sum(torch.mul(Asy, x))])

    P_world.comm.Reduce(local_results, global_results, op=MPI.SUM, root=0)

    # Because this is being computed in parallel, we risk that these norms
    # and inner products are not exactly equal, because the floating point
    # arithmetic is not commutative.  The only way to fix this is to collect
    # the results to a single rank to do the test.
    if(P_world.rank == 0):
        # Correct the norms from distributed calculation
        global_results[:4] = np.sqrt(global_results[:4])

        # Unpack the values
        norm_x, norm_y, norm_Ax, norm_Asy, ip1, ip2 = global_results

        d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
        print(f"Adjoint test: {ip1/d} {ip2/d}")
        assert(np.isclose(ip1/d, ip2/d))
    else:
        # Ranks other than 0 always pass
        pass


def test_halo_exchange_sequential():

    # Isolate a single processor to use for this test.
    if MPI.COMM_WORLD.Get_rank() == 0:
        color = 0
        comm = MPI.COMM_WORLD.Split(color)
    else:
        color = 1
        comm = MPI.COMM_WORLD.Split(color)
        return

    P_world = MPIPartition(comm)

    dims = [1, 1, 1, 1]

    P_cart = P_world.create_cartesian_subpartition(dims)

    mockup_conv_layer = MockupConvLayer()
    x_in_sizes = [1, 1, 5, 6]
    kernel_sizes = [1, 1, 3, 3]
    strides = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1, 1, 1]

    x = torch.tensor(np.random.randn(*x_in_sizes))
    y = torch.tensor(np.random.randn(*x_in_sizes))

    halo_sizes, recv_buffer_sizes, send_buffer_sizes, needed_ranges = \
        mockup_conv_layer._compute_exchange_info(x_in_sizes,
                                                 kernel_sizes,
                                                 strides,
                                                 pads,
                                                 dilations,
                                                 P_cart)

    forward_padnd_layer = PadNd(halo_sizes.astype(int), value=0)

    # Value should be random but PadNd cannot do random padding for each element. Setting
    # it to a nonzero value will be enough to guarantee correctness of the adjoint test.
    adjoint_padnd_layer = PadNd(halo_sizes.astype(int), value=1)

    x = forward_padnd_layer.forward(x)
    y = adjoint_padnd_layer.forward(y)
    x_clone = x.clone()
    y_clone = y.clone()

    ctx = DummyContext()
    halo_layer = HaloExchange(x_clone.shape, halo_sizes, recv_buffer_sizes, send_buffer_sizes, P_cart)

    Ax = HaloExchangeFunction.forward(ctx,
                                      x_clone,
                                      halo_layer.slices,
                                      halo_layer.buffers,
                                      halo_layer.neighbor_ranks,
                                      halo_layer.cartesian_partition)

    Asy = HaloExchangeFunction.backward(ctx, y_clone)[0]

    norm_x = np.sqrt((torch.norm(x)**2).numpy())
    norm_y = np.sqrt((torch.norm(y)**2).numpy())
    norm_Ax = np.sqrt((torch.norm(Ax)**2).numpy())
    norm_Asy = np.sqrt((torch.norm(Asy)**2).numpy())

    ip1 = np.array([torch.sum(torch.mul(y, Ax))])
    ip2 = np.array([torch.sum(torch.mul(Asy, x))])

    d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
    print(f"Adjoint test: {ip1/d} {ip2/d}")
    assert(np.isclose(ip1/d, ip2/d))
