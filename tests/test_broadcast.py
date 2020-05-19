def test_broadcast_parallel():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.broadcast import Broadcast
    from distdl.nn.broadcast import BroadcastFunction
    from distdl.utilities.misc import Bunch

    P_world = MPIPartition(MPI.COMM_WORLD)
    ranks = np.arange(P_world.size)

    if P_world.size > 1:
        # This should be a separate test...
        use_ranks = ranks[1:]
        P_use = P_world.create_subpartition(use_ranks)
    else:
        P_use = P_world

    tensor_sizes = np.array([7, 5])
    root_rank = 0
    layer = Broadcast(P_use, root=root_rank)

    if P_use.active:
        # Forward Input
        x = torch.Tensor(np.random.randn(*tensor_sizes))
        x_clone = x.clone()

        # Adjoint Input
        y = torch.Tensor(np.random.randn(*tensor_sizes))
        y_clone = y.clone()
    else:
        x = None
        x_clone = None
        y = None
        y_clone = None

    ctx = Bunch()

    # Apply A
    Ax = BroadcastFunction.forward(ctx, x_clone, layer.partition, layer.root)

    # Apply A*
    Asy = BroadcastFunction.backward(ctx, y_clone)[0]

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    # Compute all of the local norms and inner products.
    # We only perform the inner product calculation between
    # x and Asy on the root rank, as the input space of the forward
    # operator and the output space of the adjoint operator
    # are only relevant to the root rank
    if P_use.rank == root_rank:
        # ||x||^2
        local_results[0] = (torch.norm(x)**2).numpy()
        # ||A*@y||^2
        local_results[3] = (torch.norm(Asy)**2).numpy()
        # <A*@y, x>
        local_results[5] = np.array([torch.sum(torch.mul(Asy, x))])

    if P_use.active:
        # ||y||^2
        local_results[1] = (torch.norm(y)**2).numpy()
        # ||A@x||^2
        local_results[2] = (torch.norm(Ax)**2).numpy()
        # <A@x, y>
        local_results[4] = np.array([torch.sum(torch.mul(Ax, y))])

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
        # Ranks other than 0 always pass
        pass


def test_broadcast_sequential():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.broadcast import Broadcast
    from distdl.nn.broadcast import BroadcastFunction
    from distdl.utilities.misc import Bunch

    # Isolate a single processor to use for this test.
    if MPI.COMM_WORLD.Get_rank() == 0:
        color = 0
        comm = MPI.COMM_WORLD.Split(color)
    else:
        color = 1
        comm = MPI.COMM_WORLD.Split(color)
        return

    P_world = MPIPartition(comm)

    layer = Broadcast(P_world)

    tensor_sizes = np.array([7, 5])

    # Forward Input
    x = torch.Tensor(np.random.randn(*tensor_sizes))

    # Adjoint Input
    y = torch.Tensor(np.random.randn(*tensor_sizes))

    ctx = Bunch()

    # Apply A
    Ax = BroadcastFunction.forward(ctx, x.clone(), layer.partition, layer.root)

    # Apply A*
    Asy = BroadcastFunction.backward(ctx, y.clone())[0]

    norm_x = np.sqrt((torch.norm(x)**2).numpy())
    norm_y = np.sqrt((torch.norm(y)**2).numpy())
    norm_Ax = np.sqrt((torch.norm(Ax)**2).numpy())
    norm_Asy = np.sqrt((torch.norm(Asy)**2).numpy())

    ip1 = np.array([torch.sum(torch.mul(y, Ax))])
    ip2 = np.array([torch.sum(torch.mul(Asy, x))])

    d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
    print(f"Adjoint test: {ip1/d} {ip2/d}")
    assert(np.isclose(ip1/d, ip2/d))
