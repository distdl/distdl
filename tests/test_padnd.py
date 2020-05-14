def test_padnd():

    import numpy as np
    import torch
    from mpi4py import MPI

    from distdl.nn.padnd import PadNd
    from distdl.nn.padnd import PadNdFunction
    from distdl.utilities.misc import Bunch

    rank = MPI.COMM_WORLD.Get_rank()

    # Isolate a single processor to use for this test.
    if rank == 0:
        color = 0
        MPI.COMM_WORLD.Split(color)
    else:
        color = 1
        MPI.COMM_WORLD.Split(color)
        return

    pad_width = [(1, 2), (3, 4)]

    input_tensor_size = [7, 5]
    output_tensor_size = [s + lpad + rpad for s, (lpad, rpad) in zip(input_tensor_size, pad_width)]

    x = torch.randn(*input_tensor_size)
    y = torch.randn(*output_tensor_size)

    padnd_layer = PadNd(pad_width, value=0)

    ctx = Bunch()

    Ax = PadNdFunction.forward(ctx, x.clone(), padnd_layer.pad_width, padnd_layer.value)
    Asy = PadNdFunction.backward(ctx, y.clone())[0]

    norm_x = np.sqrt((torch.norm(x) ** 2).numpy())
    norm_y = np.sqrt((torch.norm(y) ** 2).numpy())
    norm_Ax = np.sqrt((torch.norm(Ax) ** 2).numpy())
    norm_Asy = np.sqrt((torch.norm(Asy) ** 2).numpy())

    ip1 = np.array([torch.sum(torch.mul(y, Ax))])
    ip2 = np.array([torch.sum(torch.mul(Asy, x))])

    d = np.max([norm_Ax*norm_y, norm_Asy*norm_x])
    print(f"Adjoint test: {ip1/d} {ip2/d}")
    assert(np.isclose(ip1/d, ip2/d))
