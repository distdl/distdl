from distdl.nn.halo_mixin import HaloMixin


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


class MockupPoolingLayer(HaloMixin):

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


# Tests the halo exchange function where each rank has the same padding
# (i.e. a convolutional layer)
def test_halo_exchange_function_same_padding():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.halo_exchange import HaloExchange
    from distdl.nn.halo_exchange import HaloExchangeFunction
    from distdl.nn.padnd import PadNd
    from distdl.utilities.misc import DummyContext
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P = P_world.create_partition_inclusive(np.arange(9))
    P_cart = P.create_cartesian_topology_partition([1, 1, 3, 3])

    tensor_sizes = np.array([1, 1, 10, 7])
    kernel_sizes = np.array([1, 1, 3, 3])
    strides = np.array([1, 1, 1, 1])
    pads = np.array([0, 0, 0, 0])
    dilations = np.array([1, 1, 1, 1])

    if P_cart.active:
        mockup_conv_layer = MockupConvLayer()
        halo_sizes, recv_buffer_sizes, send_buffer_sizes, _ = \
            mockup_conv_layer._compute_exchange_info(tensor_sizes,
                                                     kernel_sizes,
                                                     strides,
                                                     pads,
                                                     dilations,
                                                     P_cart.active,
                                                     P_cart.dims,
                                                     P_cart.coords)
        halo_sizes = halo_sizes.astype(int)

    else:
        halo_sizes = None
        recv_buffer_sizes = None
        send_buffer_sizes = None

    pad_layer = PadNd(halo_sizes, value=0, partition=P_cart)

    x = NoneTensor()
    if P_cart.active:
        x = torch.tensor(np.random.randn(*tensor_sizes))
    x.requires_grad = True

    x = pad_layer.forward(x)

    y = NoneTensor()
    if P_cart.active:
        y = torch.tensor(np.random.randn(*x.shape))
    y.requires_grad = True

    halo_layer = HaloExchange(x.shape, halo_sizes, recv_buffer_sizes, send_buffer_sizes, P_cart)

    x_clone = x.clone()
    y_clone = y.clone()

    ctx = DummyContext()
    Ax = HaloExchangeFunction.forward(ctx, x_clone, halo_layer.slices, halo_layer.buffers, halo_layer.neighbor_ranks, P_cart)
    Asy = HaloExchangeFunction.backward(ctx, y_clone)[0]

    x = x.detach()
    Asy = Asy.detach()
    y = y.detach()
    Ax = Ax.detach()

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    # Compute all of the local norms and inner products.
    # We only perform the inner product calculation between
    # x and Asy on the root rank, as the input space of the forward
    # operator and the output space of the adjoint operator
    # are only relevant to the root rank
    if P_cart.active:
        # ||x||^2
        local_results[0] = (torch.norm(x)**2).numpy()
        # ||y||^2
        local_results[1] = (torch.norm(y)**2).numpy()
        # ||A@x||^2
        local_results[2] = (torch.norm(Ax)**2).numpy()
        # ||A*@y||^2
        local_results[3] = (torch.norm(Asy)**2).numpy()
        # <A@x, y>
        local_results[4] = np.array([torch.sum(torch.mul(Ax, y))])
        # <A*@y, x>
        local_results[5] = np.array([torch.sum(torch.mul(Asy, x))])

    # Reduce the norms and inner products
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
        # All other ranks pass the adjoint test
        assert(True)

    # Barrier fence to ensure all enclosed MPI calls resolve.
    P_world.comm.Barrier()


# Tests the halo exchange function where each rank has different padding
# (i.e. a pooling layer)
def test_halo_exchange_function_different_padding():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.halo_exchange import HaloExchange
    from distdl.nn.halo_exchange import HaloExchangeFunction
    from distdl.nn.padnd import PadNd
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.misc import DummyContext
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P = P_world.create_partition_inclusive(np.arange(3))
    P_cart = P.create_cartesian_topology_partition([1, 1, 3])

    tensor_sizes = np.array([1, 1, 10])
    kernel_sizes = np.array([2])
    strides = np.array([2])
    pads = np.array([0])
    dilations = np.array([1])

    if P_cart.active:
        mockup_conv_layer = MockupConvLayer()
        halo_sizes, recv_buffer_sizes, send_buffer_sizes, _ = \
            mockup_conv_layer._compute_exchange_info(tensor_sizes,
                                                     kernel_sizes,
                                                     strides,
                                                     pads,
                                                     dilations,
                                                     P_cart.active,
                                                     P_cart.dims,
                                                     P_cart.coords)
        halo_sizes = halo_sizes.astype(int)
        subsizes = compute_subsizes(P_cart.dims, P_cart.cartesian_coordinates(P_cart.rank), tensor_sizes)

    else:
        halo_sizes = None
        recv_buffer_sizes = None
        send_buffer_sizes = None

    pad_layer = PadNd(halo_sizes, value=0, partition=P_cart)

    x = NoneTensor()
    if P_cart.active:
        x = torch.tensor(np.random.randn(*subsizes))
    x.requires_grad = True

    x = pad_layer.forward(x)

    y = NoneTensor()
    if P_cart.active:
        y = torch.tensor(np.random.randn(*x.shape))
    y.requires_grad = True

    halo_layer = HaloExchange(x.shape, halo_sizes, recv_buffer_sizes, send_buffer_sizes, P_cart)

    x_clone = x.clone()
    y_clone = y.clone()

    ctx = DummyContext()
    Ax = HaloExchangeFunction.forward(ctx, x_clone, halo_layer.slices, halo_layer.buffers, halo_layer.neighbor_ranks, P_cart)
    Asy = HaloExchangeFunction.backward(ctx, y_clone)[0]

    x = x.detach()
    Asy = Asy.detach()
    y = y.detach()
    Ax = Ax.detach()

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    # Compute all of the local norms and inner products.
    # We only perform the inner product calculation between
    # x and Asy on the root rank, as the input space of the forward
    # operator and the output space of the adjoint operator
    # are only relevant to the root rank
    if P_cart.active:
        # ||x||^2
        local_results[0] = (torch.norm(x)**2).numpy()
        # ||y||^2
        local_results[1] = (torch.norm(y)**2).numpy()
        # ||A@x||^2
        local_results[2] = (torch.norm(Ax)**2).numpy()
        # ||A*@y||^2
        local_results[3] = (torch.norm(Asy)**2).numpy()
        # <A@x, y>
        local_results[4] = np.array([torch.sum(torch.mul(Ax, y))])
        # <A*@y, x>
        local_results[5] = np.array([torch.sum(torch.mul(Asy, x))])

    # Reduce the norms and inner products
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
        # All other ranks pass the adjoint test
        assert(True)

    # Barrier fence to ensure all enclosed MPI calls resolve.
    P_world.comm.Barrier()
