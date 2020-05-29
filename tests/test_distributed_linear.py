def test_distributed_linear_no_bias_parallel():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.distributed_linear import DistributedLinear
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P_x = P_world.create_partition_inclusive(np.arange(4, 8))
    PC_x = P_x.create_cartesian_topology_partition([1, 4])

    P_mul = P_world.create_partition_inclusive(np.arange(0, 12))
    PC_mul = P_mul.create_cartesian_topology_partition([3, 4])

    P_y = P_world.create_partition_inclusive(np.arange(0, 3))
    PC_y = P_y.create_cartesian_topology_partition([1, 3])

    x_global_sizes = np.array([1, 12])
    y_global_sizes = np.array([1, 6])

    layer = DistributedLinear(PC_x, x_global_sizes,
                              PC_y, y_global_sizes,
                              PC_mul, bias=False)

    x = NoneTensor()
    if PC_x.active:
        x_local_subsizes = compute_subsizes(PC_x.dims, PC_x.coords, x_global_sizes)
        x = torch.Tensor(np.random.randn(*x_local_subsizes))
    x.requires_grad = True

    y = NoneTensor()
    if PC_y.active:
        # Adjoint Input
        y_local_subsizes = compute_subsizes(PC_y.dims, PC_y.coords, y_global_sizes)
        y = torch.Tensor(np.random.randn(*y_local_subsizes))

    # Apply A
    Ax = layer(x)

    # Apply A*
    Ax.backward(y)
    Asy = x.grad

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    x = x.detach()
    Asy = Asy.detach()
    y = y.detach()
    Ax = Ax.detach()

    # Compute all of the local norms and inner products.
    # We only perform the inner product calculation between
    # x and Asy on the root rank, as the input space of the forward
    # operator and the output space of the adjoint operator
    # are only relevant to the root rank
    if P_x.active:
        # ||x||^2
        local_results[0] = (torch.norm(x)**2).numpy()
        # ||A*@y||^2
        local_results[3] = (torch.norm(Asy)**2).numpy()
        # <A*@y, x>
        local_results[5] = np.array([torch.sum(torch.mul(Asy, x))])

    if P_y.active:
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

    # Test W and dW
    if PC_mul.active:

        W = layer.sublinear.weight.detach()

        dW = layer.sublinear.weight.grad.detach()

        # ||W||^2
        local_results[0] = (torch.norm(W)**2).numpy()
        # ||A*@y = dW||^2
        local_results[3] = (torch.norm(dW)**2).numpy()
        # <W, dW>
        local_results[5] = np.array([torch.sum(torch.mul(dW, W))])

    if P_y.active:
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
        norm_W, norm_y, norm_dW, norm_Asy, ip1, ip2 = global_results

        d = np.max([norm_dW*norm_y, norm_Asy*norm_W])
        print(f"Adjoint test: {ip1/d} {ip2/d}")
        assert(np.isclose(ip1/d, ip2/d))
    else:
        # All other ranks pass the adjoint test
        assert(True)

    # Barrier fence to ensure all enclosed MPI calls resolve.
    P_world.comm.Barrier()


def test_distributed_linear_bias_only_parallel():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.distributed_linear import DistributedLinear
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P_x = P_world.create_partition_inclusive(np.arange(4, 8))
    PC_x = P_x.create_cartesian_topology_partition([1, 4])

    P_mul = P_world.create_partition_inclusive(np.arange(0, 12))
    PC_mul = P_mul.create_cartesian_topology_partition([3, 4])

    P_y = P_world.create_partition_inclusive(np.arange(0, 3))
    PC_y = P_y.create_cartesian_topology_partition([1, 3])

    x_global_sizes = np.array([1, 12])
    y_global_sizes = np.array([1, 6])

    layer = DistributedLinear(PC_x, x_global_sizes,
                              PC_y, y_global_sizes,
                              PC_mul, bias=True)

    x = NoneTensor()
    if PC_x.active:
        # For this test, we are only testing to see if the adjoint works
        # correctly for the bias term.  But the adjoint test only works on the
        # Jacobian of the linear layer.  The Jacobian block for b is 0 for x
        # and W, so killing x makes the forward operator equal to its Jacobian
        # and we can test to see that adjoint is computed correctly.
        x_local_subsizes = compute_subsizes(PC_x.dims, PC_x.coords, x_global_sizes)
        x = torch.Tensor(np.zeros(x_local_subsizes))
    x.requires_grad = True

    y = NoneTensor()
    if PC_y.active:
        # Adjoint Input
        y_local_subsizes = compute_subsizes(PC_y.dims, PC_y.coords, y_global_sizes)
        y = torch.Tensor(np.random.randn(*y_local_subsizes))

    # Apply A
    Ax = layer(x)

    # Apply A*
    Ax.backward(y)

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    y = y.detach()
    Ax = Ax.detach()

    # The bias only lives in the first column of the matrix partition to
    # prevent double counting.  So, it is only defined there.
    if PC_mul.active and PC_mul.coords[-1] == 0:
        b = layer.sublinear.bias.detach()
        db = layer.sublinear.bias.grad.detach()

        # ||b||^2
        local_results[0] = (torch.norm(b)**2).numpy()
        # ||A*@y = db||^2
        local_results[3] = (torch.norm(db)**2).numpy()
        # <b, db>
        local_results[5] = np.array([torch.sum(torch.mul(db, b))])
    else:
        # If we get any other exception the test should fail, but we know
        # that bias cannot be detached here.
        try:
            b = layer.sublinear.bias.detach()
        except AttributeError:
            pass
        else:
            raise

    if P_y.active:
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
        norm_b, norm_y, norm_db, norm_Asy, ip1, ip2 = global_results

        d = np.max([norm_db*norm_y, norm_Asy*norm_b])
        print(f"Adjoint test: {ip1/d} {ip2/d}")
        assert(np.isclose(ip1/d, ip2/d))
    else:
        # All other ranks pass the adjoint test
        assert(True)

    # Barrier fence to ensure all enclosed MPI calls resolve.
    P_world.comm.Barrier()


def test_distributed_linear_no_bias_sequential():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.distributed_linear import DistributedLinear

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
    PC_x = P_world.create_cartesian_topology_partition([1, 1])
    PC_y = P_world.create_cartesian_topology_partition([1, 1])
    PC_mul = P_world.create_cartesian_topology_partition([1, 1])

    x_sizes = np.array([1, 3])
    y_sizes = np.array([1, 2])

    layer = DistributedLinear(PC_x, x_sizes, PC_y, y_sizes, PC_mul, bias=False)

    x = torch.Tensor(np.random.randn(*x_sizes))
    x.requires_grad = True
    Ax = layer(x)

    y = torch.Tensor(np.random.randn(*y_sizes))
    Ax.backward(y)
    Asy = x.grad

    x = x.detach()
    y = y.detach()
    Ax = Ax.detach()
    Asy = Asy.detach()

    norm_x = np.sqrt((torch.norm(x)**2).numpy())
    norm_y = np.sqrt((torch.norm(y)**2).numpy())
    norm_Ax = np.sqrt((torch.norm(Ax)**2).numpy())
    norm_Asy = np.sqrt((torch.norm(Asy)**2).numpy())

    ip1 = np.array([torch.sum(torch.mul(y, Ax))])
    ip2 = np.array([torch.sum(torch.mul(Asy, x))])

    d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
    print(f"Adjoint test: {ip1/d} {ip2/d}")
    assert(np.isclose(ip1/d, ip2/d))

    # Test W and dW
    W = layer.sublinear.weight.detach()
    dW = layer.sublinear.weight.grad.detach()

    norm_W = np.sqrt((torch.norm(W)**2).numpy())
    norm_dW = np.sqrt((torch.norm(dW)**2).numpy())
    ip2 = np.array([torch.sum(torch.mul(dW, W))])

    d = np.max([norm_Ax*norm_y, norm_dW*norm_W])
    print(f"Adjoint test: {ip1/d} {ip2/d}")
    assert(np.isclose(ip1/d, ip2/d))

    MPI.COMM_WORLD.Barrier()


def test_distributed_linear_bias_only_sequential():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.distributed_linear import DistributedLinear

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
    PC_x = P_world.create_cartesian_topology_partition([1, 1])
    PC_y = P_world.create_cartesian_topology_partition([1, 1])
    PC_mul = P_world.create_cartesian_topology_partition([1, 1])

    x_sizes = np.array([1, 3])
    y_sizes = np.array([1, 2])

    layer = DistributedLinear(PC_x, x_sizes, PC_y, y_sizes, PC_mul, bias=True)

    # For this test, we are only testing to see if the adjoint works
    # correctly for the bias term.  But the adjoint test only works on the
    # Jacobian of the linear layer.  The Jacobian block for b is 0 for x and
    # W, so killing x makes the forward operator equal to its Jacobian and
    # we can test to see that adjoint is computed correctly.
    x = torch.Tensor(np.zeros(x_sizes))
    x.requires_grad = True
    Ax = layer(x)

    y = torch.Tensor(np.random.randn(*y_sizes))
    Ax.backward(y)

    x = x.detach()
    y = y.detach()
    Ax = Ax.detach()

    b = layer.sublinear.bias.detach()
    db = layer.sublinear.bias.grad.detach()

    norm_b = np.sqrt((torch.norm(b)**2).numpy())
    norm_db = np.sqrt((torch.norm(db)**2).numpy())

    norm_y = np.sqrt((torch.norm(y)**2).numpy())
    norm_Ax = np.sqrt((torch.norm(Ax)**2).numpy())

    ip1 = np.array([torch.sum(torch.mul(y, Ax))])
    ip2 = np.array([torch.sum(torch.mul(db, b))])

    d = np.max([norm_Ax*norm_y, norm_db*norm_b])
    print(f"Adjoint test: {ip1/d} {ip2/d}")
    assert(np.isclose(ip1/d, ip2/d))

    MPI.COMM_WORLD.Barrier()
