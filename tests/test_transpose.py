def test_transpose_parallel():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.nn.transpose import DistributedTransposeFunction
    from distdl.utilities.misc import Bunch
    from distdl.utilities.slicing import compute_subsizes

    # Set up MPI communication wrapper
    P_world = MPIPartition(MPI.COMM_WORLD)
    ranks = np.arange(P_world.size)

    in_dims = (4, 1)
    out_dims = (2, 2)

    in_size = np.prod(in_dims)
    out_size = np.prod(out_dims)

    # In partition is the first in_size ranks
    in_ranks = ranks[:in_size]
    P_in = P_world.create_subpartition(in_ranks)
    P_in_cart = P_in.create_cartesian_subpartition(in_dims)

    # Out partition is the last outsize ranks
    out_ranks = ranks[-out_size:]
    P_out = P_world.create_subpartition(out_ranks)
    P_out_cart = P_out.create_cartesian_subpartition(out_dims)

    tensor_sizes = np.array([7, 5])
    layer = DistributedTranspose(tensor_sizes, P_in_cart, P_out_cart)

    # Forward Input
    if P_in_cart.active:
        in_subsizes = compute_subsizes(P_in_cart.comm.dims,
                                       P_in_cart.comm.Get_coords(P_in.rank),
                                       tensor_sizes)
        x = torch.Tensor(np.random.randn(*in_subsizes))
    else:
        x = None

    # Adjoint Input
    if P_out_cart.active:
        out_subsizes = compute_subsizes(P_out_cart.comm.dims,
                                        P_out_cart.comm.Get_coords(P_out.rank),
                                        tensor_sizes)
        y = torch.Tensor(np.random.randn(*out_subsizes))
    else:
        y = None

    ctx = Bunch()

    # Apply A
    x_clone = None if x is None else x.clone()
    Ax = DistributedTransposeFunction.forward(ctx, x_clone,
                                              layer.P_common, layer.sizes,
                                              layer.P_in, layer.in_data, layer.in_buffers,
                                              layer.P_out, layer.out_data, layer.out_buffers)

    # Apply A*
    y_clone = None if y is None else y.clone()
    Asy = DistributedTransposeFunction.backward(ctx, y_clone)[0]

    zero = np.zeros(1, dtype=np.float64)

    norm_x = (torch.norm(x)**2).numpy() if P_in.active else zero.copy()
    result = zero.copy()
    P_world.comm.Reduce(norm_x, result, op=MPI.SUM, root=0)
    norm_x = np.sqrt(result)

    norm_y = (torch.norm(y)**2).numpy() if P_out.active else zero.copy()
    result = zero.copy()
    P_world.comm.Reduce(norm_y, result, op=MPI.SUM, root=0)
    norm_y = np.sqrt(result)

    norm_Ax = (torch.norm(Ax)**2).numpy() if P_out.active else zero.copy()
    result = zero.copy()
    P_world.comm.Reduce(norm_Ax, result, op=MPI.SUM, root=0)
    norm_Ax = np.sqrt(result)

    norm_Asy = (torch.norm(Asy)**2).numpy() if P_in.active else zero.copy()
    result = zero.copy()
    P_world.comm.Reduce(norm_Asy, result, op=MPI.SUM, root=0)
    norm_Asy = np.sqrt(result)

    ip1 = np.array([torch.sum(torch.mul(y, Ax))]) if P_out.active else zero.copy()
    result = zero.copy()
    P_world.comm.Reduce(ip1, result, op=MPI.SUM, root=0)
    ip1[:] = result[:]

    ip2 = np.array([torch.sum(torch.mul(Asy, x))]) if P_in.active else zero.copy()
    result = zero.copy()
    P_world.comm.Reduce(ip2, result, op=MPI.SUM, root=0)
    ip2[:] = result[:]

    # Because this is being computed in parallel, we risk that these norms
    # and inner products are not exactly equal, because the floating point
    # arithmetic is not commutative.  The only way to fix this is to collect
    # the results to a single rank to do the test.
    if(P_world.rank == 0):
        d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
        print(f"Adjoint test: {ip1/d} {ip2/d}")
        assert(np.isclose(ip1/d, ip2/d))
    else:
        # Ranks other than 0 always pass
        pass


def test_transpose_sequential():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.nn.transpose import DistributedTransposeFunction
    from distdl.utilities.misc import Bunch
    from distdl.utilities.slicing import compute_subsizes

    # Isolate a single processor to use for this test.
    if MPI.COMM_WORLD.Get_rank() == 0:
        color = 0
        comm = MPI.COMM_WORLD.Split(color)
    else:
        color = 1
        comm = MPI.COMM_WORLD.Split(color)
        return

    P_world = MPIPartition(comm)

    in_dims = (1, )
    out_dims = (1, )

    P_in_cart = P_world.create_cartesian_subpartition(in_dims)
    P_out_cart = P_world.create_cartesian_subpartition(out_dims)

    tensor_sizes = np.array([7, 5])
    layer = DistributedTranspose(tensor_sizes, P_in_cart, P_out_cart)

    # Forward Input
    in_subsizes = compute_subsizes(P_in_cart.comm.dims,
                                   P_in_cart.comm.Get_coords(P_world.rank),
                                   tensor_sizes)
    x = torch.Tensor(np.random.randn(*in_subsizes))

    # Adjoint Input
    out_subsizes = compute_subsizes(P_out_cart.comm.dims,
                                    P_out_cart.comm.Get_coords(P_world.rank),
                                    tensor_sizes)
    y = torch.Tensor(np.random.randn(*out_subsizes))

    ctx = Bunch()

    # Apply A
    Ax = DistributedTransposeFunction.forward(ctx, x.clone(),
                                              layer.P_common, layer.sizes,
                                              layer.P_in, layer.in_data, layer.in_buffers,
                                              layer.P_out, layer.out_data, layer.out_buffers)

    # Apply A*
    Asy = DistributedTransposeFunction.backward(ctx, y.clone())[0]

    norm_x = np.sqrt((torch.norm(x)**2).numpy())
    norm_y = np.sqrt((torch.norm(y)**2).numpy())
    norm_Ax = np.sqrt((torch.norm(Ax)**2).numpy())
    norm_Asy = np.sqrt((torch.norm(Asy)**2).numpy())

    ip1 = np.array([torch.sum(torch.mul(y, Ax))])
    ip2 = np.array([torch.sum(torch.mul(Asy, x))])

    d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
    print(f"Adjoint test: {ip1/d} {ip2/d}")
    assert(np.isclose(ip1/d, ip2/d))
