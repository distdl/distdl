def test_transpose_parallel():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.nn.transpose import DistributedTransposeFunction
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    in_dims = (4, 1)
    out_dims = (3, 4)
    in_size = np.prod(in_dims)
    out_size = np.prod(out_dims)

    P_in = P_world.create_partition_inclusive(np.arange(0, in_size))
    PC_in = P_in.create_cartesian_topology_partition(in_dims)

    P_out = P_world.create_partition_inclusive(np.arange(P_world.size-out_size, P_world.size))
    PC_out = P_out.create_cartesian_topology_partition(out_dims)

    global_tensor_sizes = np.array([77, 55])

    layer = DistributedTranspose(global_tensor_sizes, PC_in, PC_out)

    # Forward Input
    x = NoneTensor()
    if PC_in.active:
        in_subsizes = compute_subsizes(PC_in.comm.dims,
                                       PC_in.comm.Get_coords(P_in.rank),
                                       global_tensor_sizes)
        x = torch.Tensor(np.random.randn(*in_subsizes))
    x.requires_grad = True

    # Adjoint Input
    y = NoneTensor()
    if PC_out.active:
        out_subsizes = compute_subsizes(PC_out.comm.dims,
                                        PC_out.comm.Get_coords(P_out.rank),
                                        global_tensor_sizes)
        y = torch.Tensor(np.random.randn(*out_subsizes))

    ctx = DistributedTransposeFunction()

    # Apply A
    Ax = DistributedTransposeFunction.forward(ctx, x,
                                              layer.P_union,
                                              layer.sizes,
                                              layer.P_in,
                                              layer.in_data,
                                              layer.in_buffers,
                                              layer.P_out,
                                              layer.out_data,
                                              layer.out_buffers,
                                              layer.dtype)

    # Apply A*
    Asy = DistributedTransposeFunction.backward(ctx, y)[0]

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
    # assert(0)

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


def test_transpose_as_scatter():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.nn.transpose import DistributedTransposeFunction
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    in_dims = (1,)
    out_dims = (4, 3)
    in_size = np.prod(in_dims)
    out_size = np.prod(out_dims)

    P_in = P_world.create_partition_inclusive(np.arange(0, in_size))
    PC_in = P_in.create_cartesian_topology_partition(in_dims)

    P_out = P_world.create_partition_inclusive(np.arange(P_world.size-out_size, P_world.size))
    PC_out = P_out.create_cartesian_topology_partition(out_dims)

    global_tensor_sizes = np.array([77, 55])

    layer = DistributedTranspose(global_tensor_sizes, PC_in, PC_out)

    # Forward Input
    x = NoneTensor()
    if PC_in.active:
        in_subsizes = compute_subsizes(PC_in.comm.dims,
                                       PC_in.comm.Get_coords(P_in.rank),
                                       global_tensor_sizes)
        x = torch.Tensor(np.random.randn(*in_subsizes))
    x.requires_grad = True

    # Adjoint Input
    y = NoneTensor()
    if PC_out.active:
        out_subsizes = compute_subsizes(PC_out.comm.dims,
                                        PC_out.comm.Get_coords(P_out.rank),
                                        global_tensor_sizes)
        y = torch.Tensor(np.random.randn(*out_subsizes))

    ctx = DistributedTransposeFunction()

    # Apply A
    Ax = DistributedTransposeFunction.forward(ctx, x,
                                              layer.P_union,
                                              layer.sizes,
                                              layer.P_in,
                                              layer.in_data,
                                              layer.in_buffers,
                                              layer.P_out,
                                              layer.out_data,
                                              layer.out_buffers,
                                              layer.dtype)

    # Apply A*
    Asy = DistributedTransposeFunction.backward(ctx, y)[0]

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
    # assert(0)

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


def test_transpose_as_gather():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.transpose import DistributedTranspose
    from distdl.nn.transpose import DistributedTransposeFunction
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    in_dims = (3, 4)
    out_dims = (1,)
    in_size = np.prod(in_dims)
    out_size = np.prod(out_dims)

    P_in = P_world.create_partition_inclusive(np.arange(0, in_size))
    PC_in = P_in.create_cartesian_topology_partition(in_dims)

    P_out = P_world.create_partition_inclusive(np.arange(P_world.size-out_size, P_world.size))
    PC_out = P_out.create_cartesian_topology_partition(out_dims)

    global_tensor_sizes = np.array([77, 55])

    layer = DistributedTranspose(global_tensor_sizes, PC_in, PC_out)

    # Forward Input
    x = NoneTensor()
    if PC_in.active:
        in_subsizes = compute_subsizes(PC_in.comm.dims,
                                       PC_in.comm.Get_coords(P_in.rank),
                                       global_tensor_sizes)
        x = torch.Tensor(np.random.randn(*in_subsizes))
    x.requires_grad = True

    # Adjoint Input
    y = NoneTensor()
    if PC_out.active:
        out_subsizes = compute_subsizes(PC_out.comm.dims,
                                        PC_out.comm.Get_coords(P_out.rank),
                                        global_tensor_sizes)
        y = torch.Tensor(np.random.randn(*out_subsizes))

    ctx = DistributedTransposeFunction()

    # Apply A
    Ax = DistributedTransposeFunction.forward(ctx, x,
                                              layer.P_union,
                                              layer.sizes,
                                              layer.P_in,
                                              layer.in_data,
                                              layer.in_buffers,
                                              layer.P_out,
                                              layer.out_data,
                                              layer.out_buffers,
                                              layer.dtype)

    # Apply A*
    Asy = DistributedTransposeFunction.backward(ctx, y)[0]

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
    # assert(0)

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
