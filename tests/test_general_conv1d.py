def test_conv_1d_no_bias_co2_ci1_parallel():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.general_conv import DistributedGeneralConv1d
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P_3 = P_world.create_partition_inclusive(np.arange(0, 3))
    P_x = P_3.create_cartesian_topology_partition([1, 1, 3])

    P_6 = P_world.create_partition_inclusive(np.arange(0, 6))
    P_y = P_6.create_cartesian_topology_partition([1, 2, 3])
    P_w = P_6.create_cartesian_topology_partition([2, 1, 3])

    global_tensor_sizes = np.array([1, 5, 10])

    layer = DistributedGeneralConv1d(global_tensor_sizes,
                                     P_x, P_y, P_w,
                                     in_channels=global_tensor_sizes[1],
                                     out_channels=10,
                                     kernel_size=[3], bias=False)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              global_tensor_sizes)
        x = torch.Tensor(np.random.randn(*input_tensor_sizes))
    x.requires_grad = True

    Ax = layer(x)

    y = NoneTensor()
    if P_y.active:
        y = torch.Tensor(np.random.randn(*Ax.shape))
    y.requires_grad = True

    Ax.backward(y)
    Asy = x.grad

    x = x.detach()
    Ax = Ax.detach()
    y = y.detach()
    Asy = Asy.detach()

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

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

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    if P_w.active:

        W = layer._weight.detach()
        dW = layer._weight.grad.detach()

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

def test_conv_1d_no_bias_co1_ci2_parallel():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.general_conv import DistributedGeneralConv1d
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P_3 = P_world.create_partition_inclusive(np.arange(0, 3))
    P_6 = P_world.create_partition_inclusive(np.arange(0, 6))

    P_x = P_6.create_cartesian_topology_partition([1, 2, 3])
    P_y = P_3.create_cartesian_topology_partition([1, 1, 3])
    P_w = P_6.create_cartesian_topology_partition([1, 2, 3])

    global_tensor_sizes = np.array([1, 5, 10])

    layer = DistributedGeneralConv1d(global_tensor_sizes,
                                     P_x, P_y, P_w,
                                     in_channels=global_tensor_sizes[1],
                                     out_channels=10,
                                     kernel_size=[3], bias=False)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              global_tensor_sizes)
        x = torch.Tensor(np.random.randn(*input_tensor_sizes))
    x.requires_grad = True

    Ax = layer(x)

    y = NoneTensor()
    if P_y.active:
        y = torch.Tensor(np.random.randn(*Ax.shape))
    y.requires_grad = True

    Ax.backward(y)
    Asy = x.grad

    x = x.detach()
    Ax = Ax.detach()
    y = y.detach()
    Asy = Asy.detach()

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

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

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    if P_w.active:

        W = layer._weight.detach()
        dW = layer._weight.grad.detach()

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

def test_conv_1d_no_bias_co2_ci2_parallel():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.general_conv import DistributedGeneralConv1d
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P_4 = P_world.create_partition_inclusive(np.arange(0, 4))
    P_8 = P_world.create_partition_inclusive(np.arange(0, 8))

    P_x = P_4.create_cartesian_topology_partition([1, 2, 2])
    P_y = P_4.create_cartesian_topology_partition([1, 2, 2])
    P_w = P_8.create_cartesian_topology_partition([2, 2, 2])

    global_tensor_sizes = np.array([1, 5, 10])

    layer = DistributedGeneralConv1d(global_tensor_sizes,
                                     P_x, P_y, P_w,
                                     in_channels=global_tensor_sizes[1],
                                     out_channels=10,
                                     kernel_size=[3], bias=False)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              global_tensor_sizes)
        x = torch.Tensor(np.random.randn(*input_tensor_sizes))
    x.requires_grad = True

    Ax = layer(x)

    y = NoneTensor()
    if P_y.active:
        y = torch.Tensor(np.random.randn(*Ax.shape))
    y.requires_grad = True

    Ax.backward(y)
    Asy = x.grad

    x = x.detach()
    Ax = Ax.detach()
    y = y.detach()
    Asy = Asy.detach()

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

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

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    if P_w.active:

        W = layer._weight.detach()
        dW = layer._weight.grad.detach()

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

def test_conv_1d_bias_only_co2_ci1_parallel():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.general_conv import DistributedGeneralConv1d
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P_3 = P_world.create_partition_inclusive(np.arange(0, 3))
    P_x = P_3.create_cartesian_topology_partition([1, 1, 3])

    P_6 = P_world.create_partition_inclusive(np.arange(0, 6))
    P_y = P_6.create_cartesian_topology_partition([1, 2, 3])
    P_w = P_6.create_cartesian_topology_partition([2, 1, 3])

    global_tensor_sizes = np.array([1, 5, 10])

    layer = DistributedGeneralConv1d(global_tensor_sizes,
                                     P_x, P_y, P_w,
                                     in_channels=global_tensor_sizes[1],
                                     out_channels=10,
                                     kernel_size=[3], bias=True)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              global_tensor_sizes)
        x = torch.zeros(*input_tensor_sizes)
    x.requires_grad = True

    Ax = layer(x)

    y = NoneTensor()
    if P_y.active:
        y = torch.Tensor(np.random.randn(*Ax.shape))
    y.requires_grad = True

    Ax.backward(y)

    y = y.detach()
    Ax = Ax.detach()

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    if P_w.active and layer.stores_bias:

        b = layer._bias.detach()
        db = layer._bias.grad.detach()

        # ||b||^2
        local_results[0] = (torch.norm(b)**2).numpy()
        # ||A*@y = db||^2
        local_results[3] = (torch.norm(db)**2).numpy()
        # <b, db>
        local_results[5] = np.array([torch.sum(torch.mul(db, b))])

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

def test_conv_1d_bias_only_co1_ci2_parallel():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.general_conv import DistributedGeneralConv1d
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P_3 = P_world.create_partition_inclusive(np.arange(0, 3))
    P_6 = P_world.create_partition_inclusive(np.arange(0, 6))

    P_x = P_6.create_cartesian_topology_partition([1, 2, 3])
    P_y = P_3.create_cartesian_topology_partition([1, 1, 3])
    P_w = P_6.create_cartesian_topology_partition([1, 2, 3])

    global_tensor_sizes = np.array([1, 5, 10])

    layer = DistributedGeneralConv1d(global_tensor_sizes,
                                     P_x, P_y, P_w,
                                     in_channels=global_tensor_sizes[1],
                                     out_channels=10,
                                     kernel_size=[3], bias=True)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              global_tensor_sizes)
        x = torch.zeros(*input_tensor_sizes)
    x.requires_grad = True

    Ax = layer(x)

    y = NoneTensor()
    if P_y.active:
        y = torch.Tensor(np.random.randn(*Ax.shape))
    y.requires_grad = True

    Ax.backward(y)

    y = y.detach()
    Ax = Ax.detach()

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    if P_w.active and layer.stores_bias:

        b = layer._bias.detach()
        db = layer._bias.grad.detach()

        # ||b||^2
        local_results[0] = (torch.norm(b)**2).numpy()
        # ||A*@y = db||^2
        local_results[3] = (torch.norm(db)**2).numpy()
        # <b, db>
        local_results[5] = np.array([torch.sum(torch.mul(db, b))])

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

def test_conv_1d_bias_only_co2_ci2_parallel():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.general_conv import DistributedGeneralConv1d
    from distdl.utilities.slicing import compute_subsizes
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P_4 = P_world.create_partition_inclusive(np.arange(0, 4))
    P_8 = P_world.create_partition_inclusive(np.arange(0, 8))

    P_x = P_4.create_cartesian_topology_partition([1, 2, 2])
    P_y = P_4.create_cartesian_topology_partition([1, 2, 2])
    P_w = P_8.create_cartesian_topology_partition([2, 2, 2])

    global_tensor_sizes = np.array([1, 5, 10])

    layer = DistributedGeneralConv1d(global_tensor_sizes,
                                     P_x, P_y, P_w,
                                     in_channels=global_tensor_sizes[1],
                                     out_channels=10,
                                     kernel_size=[3], bias=True)

    x = NoneTensor()
    if P_x.active:
        input_tensor_sizes = compute_subsizes(P_x.dims,
                                              P_x.coords,
                                              global_tensor_sizes)
        x = torch.zeros(*input_tensor_sizes)
    x.requires_grad = True

    Ax = layer(x)

    y = NoneTensor()
    if P_y.active:
        y = torch.Tensor(np.random.randn(*Ax.shape))
    y.requires_grad = True

    Ax.backward(y)

    y = y.detach()
    Ax = Ax.detach()

    local_results = np.zeros(6, dtype=np.float64)
    global_results = np.zeros(6, dtype=np.float64)

    if P_w.active and layer.stores_bias:

        b = layer._bias.detach()
        db = layer._bias.grad.detach()

        # ||b||^2
        local_results[0] = (torch.norm(b)**2).numpy()
        # ||A*@y = db||^2
        local_results[3] = (torch.norm(db)**2).numpy()
        # <b, db>
        local_results[5] = np.array([torch.sum(torch.mul(db, b))])

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
