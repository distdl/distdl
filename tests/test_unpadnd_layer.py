def test_unpadnd_layer_postive_padding():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.unpadnd import UnPadNd
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()
    P = P_world.create_partition_inclusive(np.arange(1))

    tensor_sizes = np.array([4, 5, 6])
    pads = np.array([[1, 2], [1, 2], [1, 2]])

    unpadnd_layer = UnPadNd(pads, value=0, partition=P)

    x = NoneTensor()
    if P.active:
        x = torch.tensor(np.random.randn(*tensor_sizes))
    x.requires_grad = True

    Ax = unpadnd_layer(x)

    y = NoneTensor()
    if P.active:
        y = torch.tensor(np.random.randn(*Ax.shape))
    y.requires_grad = True

    Ax.backward(y)
    Asy = x.grad

    x = x.detach()
    Asy = Asy.detach()
    y = y.detach()
    Ax = Ax.detach()

    if P.active:
        norm_x = np.sqrt((torch.norm(x)**2).numpy())
        norm_y = np.sqrt((torch.norm(y)**2).numpy())
        norm_Ax = np.sqrt((torch.norm(Ax)**2).numpy())
        norm_Asy = np.sqrt((torch.norm(Asy)**2).numpy())
        ip1 = np.array([torch.sum(torch.mul(Ax, y))])
        ip2 = np.array([torch.sum(torch.mul(Asy, x))])

        d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
        print(f"Adjoint test: {ip1/d} {ip2/d}")
        assert(np.isclose(ip1/d, ip2/d))

    else:
        assert(True)


# Tests unpadnd layer when each dimension has nonnegative padding
def test_unpadnd_layer_nonnegative_padding():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.unpadnd import UnPadNd
    from distdl.utilities.torch import NoneTensor

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()
    P = P_world.create_partition_inclusive(np.arange(1))

    tensor_sizes = np.array([3, 4, 5])
    pads = np.array([[1, 0], [0, 2], [0, 0]])

    unpadnd_layer = UnPadNd(pads, value=0, partition=P)

    x = NoneTensor()
    if P.active:
        x = torch.tensor(np.random.randn(*tensor_sizes))
    x.requires_grad = True

    Ax = unpadnd_layer(x)

    y = NoneTensor()
    if P.active:
        y = torch.tensor(np.random.randn(*Ax.shape))
    y.requires_grad = True

    Ax.backward(y)
    Asy = x.grad

    x = x.detach()
    Asy = Asy.detach()
    y = y.detach()
    Ax = Ax.detach()

    if P.active:
        norm_x = np.sqrt((torch.norm(x)**2).numpy())
        norm_y = np.sqrt((torch.norm(y)**2).numpy())
        norm_Ax = np.sqrt((torch.norm(Ax)**2).numpy())
        norm_Asy = np.sqrt((torch.norm(Asy)**2).numpy())
        ip1 = np.array([torch.sum(torch.mul(Ax, y))])
        ip2 = np.array([torch.sum(torch.mul(Asy, x))])

        d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
        print(f"Adjoint test: {ip1/d} {ip2/d}")
        assert(np.isclose(ip1/d, ip2/d))

    else:
        assert(True)
