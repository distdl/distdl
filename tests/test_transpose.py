def test_transpose_parallel():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.nn.transpose import DistributedTranspose
    from distdl.nn.transpose import DistributedTransposeFunction
    from distdl.utilities.misc import Bunch
    from distdl.utilities.slicing import compute_subsizes

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    in_dims = (4, 1)
    in_comm = comm.Create_cart(dims=in_dims)

    in_rank = in_comm.Get_rank()

    out_dims = (2, 2)
    out_comm = comm.Create_cart(dims=out_dims)

    out_rank = out_comm.Get_rank()

    sizes = np.array([7, 5])

    in_subsizes = compute_subsizes(in_comm.dims,
                                   in_comm.Get_coords(in_rank),
                                   sizes)

    out_subsizes = compute_subsizes(out_comm.dims,
                                    out_comm.Get_coords(out_rank),
                                    sizes)

    layer = DistributedTranspose(sizes, comm, in_comm, out_comm)

    # Forward Input
    x = torch.Tensor(np.random.randn(*in_subsizes))

    # Adjoint Input
    y = torch.Tensor(np.random.randn(*out_subsizes))

    ctx = Bunch()

    # Apply A
    Ax = DistributedTransposeFunction.forward(ctx, x.clone(), layer.parent_comm, layer.sizes,
                                              layer.in_slices, layer.in_buffers, layer.in_comm,
                                              layer.out_slices, layer.out_buffers, layer.out_comm)

    # Apply A*
    Asy = DistributedTransposeFunction.backward(ctx, y.clone())[0]

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


def test_transpose_sequential():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.nn.transpose import DistributedTranspose
    from distdl.nn.transpose import DistributedTransposeFunction
    from distdl.utilities.misc import Bunch
    from distdl.utilities.slicing import compute_subsizes

    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()

    # Isolate a single processor to use for this test.
    if rank == 0:
        color = 0
        comm = MPI.COMM_WORLD.Split(color)
    else:
        color = 1
        comm = MPI.COMM_WORLD.Split(color)
        return

    in_dims = (1, )
    in_comm = comm.Create_cart(dims=in_dims)

    in_rank = in_comm.Get_rank()

    out_dims = (1, )
    out_comm = comm.Create_cart(dims=out_dims)

    out_rank = out_comm.Get_rank()

    sizes = np.array([7, 5])

    in_subsizes = compute_subsizes(in_comm.dims,
                                   in_comm.Get_coords(in_rank),
                                   sizes)

    out_subsizes = compute_subsizes(out_comm.dims,
                                    out_comm.Get_coords(out_rank),
                                    sizes)

    layer = DistributedTranspose(sizes, comm, in_comm, out_comm)

    # Forward Input
    x = torch.Tensor(np.random.randn(*in_subsizes))

    # Adjoint Input
    y = torch.Tensor(np.random.randn(*out_subsizes))

    ctx = Bunch()

    # Apply A
    Ax = DistributedTransposeFunction.forward(ctx, x.clone(), layer.parent_comm, layer.sizes,
                                              layer.in_slices, layer.in_buffers, layer.in_comm,
                                              layer.out_slices, layer.out_buffers, layer.out_comm)

    # Apply A*
    Asy = DistributedTransposeFunction.backward(ctx, y.clone())[0]

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

    d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
    print(f"Adjoint test: {ip1/d} {ip2/d}")
    assert(np.isclose(ip1/d, ip2/d))
