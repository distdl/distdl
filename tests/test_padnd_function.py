# Tests padnd layer when each dimension has positive padding
def test_padnd_layer_postive_padding():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.padnd import PadNd
    from distdl.nn.padnd import PadNdFunction
    from distdl.utilities.torch import NoneTensor
    from distdl.utilities.misc import DummyContext

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()
    P = P_world.create_partition_inclusive(np.arange(1))

    tensor_sizes = np.array([3, 4, 5])
    pads = np.array([[1, 2], [1, 2], [1, 2]])
    padded_sizes = [t + lpad + rpad for t, (lpad, rpad) in zip(tensor_sizes, pads)]

    padnd_layer = PadNd(pads, value=0, partition=P)

    x = NoneTensor()
    if P.active:
        x = torch.tensor(np.random.randn(*tensor_sizes))
    x.requires_grad = True

    y = NoneTensor()
    if P.active:
        y = torch.tensor(np.random.randn(*padded_sizes))
    y.requires_grad = True

    ctx = DummyContext()
    Ax = PadNdFunction.forward(ctx, x, padnd_layer.pad_width, padnd_layer.value, padnd_layer.partition)
    Asy = PadNdFunction.backward(ctx, y)[0]

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


# Tests padnd layer when each dimension has nonnegative padding
def test_padnd_layer_nonnegative_padding():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.padnd import PadNd
    from distdl.nn.padnd import PadNdFunction
    from distdl.utilities.torch import NoneTensor
    from distdl.utilities.misc import DummyContext

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()
    P = P_world.create_partition_inclusive(np.arange(1))

    tensor_sizes = np.array([3, 4, 5])
    pads = np.array([[1, 0], [0, 2], [0, 0]])
    padded_sizes = [t + lpad + rpad for t, (lpad, rpad) in zip(tensor_sizes, pads)]

    padnd_layer = PadNd(pads, value=0, partition=P)

    x = NoneTensor()
    if P.active:
        x = torch.tensor(np.random.randn(*tensor_sizes))
    x.requires_grad = True

    y = NoneTensor()
    if P.active:
        y = torch.tensor(np.random.randn(*padded_sizes))
    y.requires_grad = True

    ctx = DummyContext()
    Ax = PadNdFunction.forward(ctx, x, padnd_layer.pad_width, padnd_layer.value, padnd_layer.partition)
    Asy = PadNdFunction.backward(ctx, y)[0]

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
