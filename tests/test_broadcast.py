
# Tests if root in P_in is also in P_out
def test_broadcast_parallel_overlap():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.newnew_partition import MPIPartition
    from distdl.nn.broadcast import Broadcast
    from distdl.nn.broadcast import BroadcastFunction

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P_in = P_world.create_partition_inclusive(np.arange(4, 8))
    PC_in = P_in.create_cartesian_topology_partition([2, 2, 1])

    P_out = P_world.create_partition_inclusive(np.arange(0, 12))
    PC_out = P_out.create_cartesian_topology_partition([2, 2, 3])

    layer = Broadcast(PC_in, PC_out)

    tensor_sizes = np.array([7, 5])

    x = None
    x_clone = None
    if PC_in.active:

        x = torch.Tensor(np.random.randn(*tensor_sizes))
        x_clone = x.clone()

    y = None
    y_clone = None
    if PC_out.active:
        # Adjoint Input
        y = torch.Tensor(np.random.randn(*tensor_sizes))
        y_clone = y.clone()

    ctx = BroadcastFunction()

    # Apply A
    Ax = BroadcastFunction.forward(ctx, x_clone,
                                   layer.P_bcast_same,
                                   layer.P_bcast_send,
                                   layer.P_bcast_recv,
                                   layer.dtype)

    # Apply A*
    Asy = BroadcastFunction.backward(ctx, y_clone)[0]

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    # Compute all of the local norms and inner products.
    # We only perform the inner product calculation between
    # x and Asy on the root rank, as the input space of the forward
    # operator and the output space of the adjoint operator
    # are only relevant to the root rank
    if P_in.active:
        # ||x||^2
        local_results[0] = (torch.norm(x)**2).numpy()
        # ||A*@y||^2
        local_results[3] = (torch.norm(Asy)**2).numpy()
        # <A*@y, x>
        local_results[5] = np.array([torch.sum(torch.mul(Asy, x))])

    if P_out.active:
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
        # All other ranks pass the adjoint test
        assert(True)

    # Barrier fence to ensure all enclosed MPI calls resolve.
    P_world.comm.Barrier()

# Tests if all ranks have work to do, but root of P_in is not in P_out
def test_broadcast_parallel_barely_disjoint():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.broadcast import Broadcast
    from distdl.nn.broadcast import BroadcastFunction

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P_in = P_world.create_partition_inclusive(np.arange(0, 4))
    PC_in = P_in.create_cartesian_topology_partition([2, 2, 1])

    P_out = P_world.create_partition_inclusive(np.arange(4, 16))
    PC_out = P_out.create_cartesian_topology_partition([2, 2, 3])

    layer = Broadcast(PC_in, PC_out)

    tensor_sizes = np.array([7, 5])

    x = None
    x_clone = None
    if PC_in.active:

        x = torch.Tensor(np.random.randn(*tensor_sizes))
        x_clone = x.clone()

    y = None
    y_clone = None
    if PC_out.active:
        # Adjoint Input
        y = torch.Tensor(np.random.randn(*tensor_sizes))
        y_clone = y.clone()

    ctx = BroadcastFunction()

    # Apply A
    Ax = BroadcastFunction.forward(ctx, x_clone,
                                   layer.P_bcast_same,
                                   layer.P_bcast_send,
                                   layer.P_bcast_recv,
                                   layer.dtype)

    # Apply A*
    Asy = BroadcastFunction.backward(ctx, y_clone)[0]

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    # Compute all of the local norms and inner products.
    # We only perform the inner product calculation between
    # x and Asy on the root rank, as the input space of the forward
    # operator and the output space of the adjoint operator
    # are only relevant to the root rank
    if P_in.active:
        # ||x||^2
        local_results[0] = (torch.norm(x)**2).numpy()
        # ||A*@y||^2
        local_results[3] = (torch.norm(Asy)**2).numpy()
        # <A*@y, x>
        local_results[5] = np.array([torch.sum(torch.mul(Asy, x))])

    if P_out.active:
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
        # All other ranks pass the adjoint test
        assert(True)

    # Barrier fence to ensure all enclosed MPI calls resolve.
    P_world.comm.Barrier()


# Tests if there are ranks that do not have any work to do at all but still
# go through the layer logic.
def test_broadcast_parallel_completely_disjoint():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.broadcast import Broadcast
    from distdl.nn.broadcast import BroadcastFunction

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P_in = P_world.create_partition_inclusive(np.arange(0, 4))
    PC_in = P_in.create_cartesian_topology_partition([2, 2, 1])

    P_out = P_world.create_partition_inclusive(np.arange(5, 17))
    PC_out = P_out.create_cartesian_topology_partition([2, 2, 3])

    layer = Broadcast(PC_in, PC_out)

    tensor_sizes = np.array([7, 5])

    x = None
    x_clone = None
    if PC_in.active:

        x = torch.Tensor(np.random.randn(*tensor_sizes))
        x_clone = x.clone()

    y = None
    y_clone = None
    if PC_out.active:
        # Adjoint Input
        y = torch.Tensor(np.random.randn(*tensor_sizes))
        y_clone = y.clone()

    ctx = BroadcastFunction()

    # Apply A
    Ax = BroadcastFunction.forward(ctx, x_clone,
                                   layer.P_bcast_same,
                                   layer.P_bcast_send,
                                   layer.P_bcast_recv,
                                   layer.dtype)

    # Apply A*
    Asy = BroadcastFunction.backward(ctx, y_clone)[0]

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    # Compute all of the local norms and inner products.
    # We only perform the inner product calculation between
    # x and Asy on the root rank, as the input space of the forward
    # operator and the output space of the adjoint operator
    # are only relevant to the root rank
    if P_in.active:
        # ||x||^2
        local_results[0] = (torch.norm(x)**2).numpy()
        # ||A*@y||^2
        local_results[3] = (torch.norm(Asy)**2).numpy()
        # <A*@y, x>
        local_results[5] = np.array([torch.sum(torch.mul(Asy, x))])

    if P_out.active:
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
        # All other ranks pass the adjoint test
        assert(True)

    # Barrier fence to ensure all enclosed MPI calls resolve.
    P_world.comm.Barrier()


def test_broadcast_sequential():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.broadcast import Broadcast

    MPI.COMM_WORLD.Barrier()

    # Isolate a single processor to use for this test.
    if MPI.COMM_WORLD.Get_rank() == 0:
        color = 0
        comm = MPI.COMM_WORLD.Split(color)
    else:
        color = 1
        comm = MPI.COMM_WORLD.Split(color)

        MPI.COMM_WORLD.Barrier()
        return

    P_world = MPIPartition(comm)

    layer = Broadcast(P_world, P_world)

    tensor_sizes = np.array([7, 5])

    # Forward Input
    x = torch.Tensor(np.random.randn(*tensor_sizes))
    x.requires_grad = True

    # Adjoint Input
    y = torch.Tensor(np.random.randn(*tensor_sizes))

    # Apply A
    Ax = layer.forward(x)

    # Apply A*
    Ax.backward(y)
    Asy = x.grad

    x_d = x.detach()
    y_d = y.detach()
    Ax_d = Ax.detach()
    Asy_d = Asy.detach()

    norm_x = np.sqrt((torch.norm(x_d)**2).numpy())
    norm_y = np.sqrt((torch.norm(y_d)**2).numpy())
    norm_Ax = np.sqrt((torch.norm(Ax_d)**2).numpy())
    norm_Asy = np.sqrt((torch.norm(Asy_d)**2).numpy())

    ip1 = np.array([torch.sum(torch.mul(y_d, Ax_d))])
    ip2 = np.array([torch.sum(torch.mul(Asy_d, x_d))])

    d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
    print(f"Adjoint test: {ip1/d} {ip2/d}")
    assert(np.isclose(ip1/d, ip2/d))

    MPI.COMM_WORLD.Barrier()
