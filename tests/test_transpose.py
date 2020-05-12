def test_transpose():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.nn.transpose import DistributedTranspose
    from distdl.nn.transpose import DistributedTransposeFunction
    from distdl.utilities.debug import print_sequential
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
    in_buffers, out_buffers = layer._allocate_buffers(np.float64)

    # Forward Input
    x = torch.Tensor(np.random.randn(*in_subsizes))

    # Adjoint Input
    y = torch.Tensor(np.random.randn(*out_subsizes))

    ctx = Bunch()

    # Apply A
    Ax = DistributedTransposeFunction.forward(ctx, x.clone(), comm, sizes,
                                              layer.in_slices, in_buffers, in_comm,
                                              layer.out_slices, out_buffers, out_comm)

    # Apply A*
    Asy = DistributedTransposeFunction.backward(ctx, y.clone())[0]

    norm_x = torch.norm(x)**2
    comm.reduce(norm_x, op=MPI.SUM, root=0)
    norm_x = np.sqrt(norm_x)

    norm_y = torch.norm(y)**2
    comm.reduce(norm_y, op=MPI.SUM, root=0)
    norm_y = np.sqrt(norm_y)

    norm_Ax = torch.norm(Ax)**2
    comm.reduce(norm_Ax, op=MPI.SUM, root=0)
    norm_Ax = np.sqrt(norm_Ax)

    norm_Asy = torch.norm(Asy)**2
    comm.reduce(norm_Asy, op=MPI.SUM, root=0)
    norm_Asy = np.sqrt(norm_Asy)

    ip1 = torch.sum(torch.mul(y, Ax))
    comm.reduce(ip1, op=MPI.SUM, root=0)

    ip2 = torch.sum(torch.mul(Asy, x))
    comm.reduce(ip2, op=MPI.SUM, root=0)
    print_sequential(comm, f'{rank}: {ip1} {ip2}')

    if(rank == 0):
        e = abs(ip1 - ip2) / np.max([norm_Ax*norm_y, norm_Asy*norm_x])
        print(f"Adjoint test: {e}")
