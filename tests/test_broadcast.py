def test_broadcast_parallel():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.nn.broadcast import Broadcast
    from distdl.nn.broadcast import BroadcastFunction
    from distdl.utilities.misc import Bunch

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    layer = Broadcast(comm=comm, root=0)

    tensor_size = [7, 5]

    # Forward Input
    x = torch.Tensor(np.random.randn(*tensor_size))

    # Adjoint Input
    y = torch.Tensor(np.random.randn(*tensor_size))

    ctx = Bunch()

    # Apply A
    Ax = BroadcastFunction.forward(ctx, x.clone(), layer.comm, layer.root)

    # Apply A*
    Asy = BroadcastFunction.backward(ctx, y.clone())[0]

    # We only perform the inner product calculation between
    # x and Asy on rank 0, as the input space of the forward
    # operator and the output space of the adjoint operator
    # are only relevant to the root rank
    if rank == 0:
        norm_x = np.sqrt((torch.norm(x) ** 2).numpy())
        norm_Asy = np.sqrt((torch.norm(x) ** 2).numpy())
        ip2 = np.array([torch.sum(torch.mul(Asy, x))])

    norm_y = (torch.norm(y)**2).numpy()
    result = np.array([0.0], dtype=norm_y.dtype)
    comm.Reduce(norm_y, result, op=MPI.SUM, root=0)
    norm_y = np.sqrt(result)

    norm_Ax = (torch.norm(Ax)**2).numpy()
    result = np.array([0.0], dtype=norm_Ax.dtype)
    comm.Reduce(norm_Ax, result, op=MPI.SUM, root=0)
    norm_Ax = np.sqrt(result)

    ip1 = np.array([torch.sum(torch.mul(y, Ax))])
    result = np.array([0.0], dtype=ip1.dtype)
    comm.Reduce(ip1, result, op=MPI.SUM, root=0)
    ip1[:] = result[:]

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


def test_broadcast_sequential():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.nn.broadcast import Broadcast
    from distdl.nn.broadcast import BroadcastFunction
    from distdl.utilities.misc import Bunch

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Isolate a single processor to use for this test.
    if rank == 0:
        color = 0
        comm = MPI.COMM_WORLD.Split(color)
    else:
        color = 1
        comm = MPI.COMM_WORLD.Split(color)
        return

    layer = Broadcast(comm=comm, root=0)

    tensor_size = [7, 5]

    # Forward Input
    x = torch.Tensor(np.random.randn(*tensor_size))

    # Adjoint Input
    y = torch.Tensor(np.random.randn(*tensor_size))

    ctx = Bunch()

    # Apply A
    Ax = BroadcastFunction.forward(ctx, x.clone(), layer.comm, layer.root)

    # Apply A*
    Asy = BroadcastFunction.backward(ctx, y.clone())[0]

    norm_x = np.sqrt((torch.norm(x) ** 2).numpy())
    norm_Asy = np.sqrt((torch.norm(x) ** 2).numpy())
    ip2 = np.array([torch.sum(torch.mul(Asy, x))])

    norm_y = np.sqrt((torch.norm(y)**2).numpy())
    norm_Ax = np.sqrt((torch.norm(Ax)**2).numpy())
    ip1 = np.array([torch.sum(torch.mul(y, Ax))])

    d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
    print(f"Adjoint test: {ip1/d} {ip2/d}")
    assert(np.isclose(ip1/d, ip2/d))
