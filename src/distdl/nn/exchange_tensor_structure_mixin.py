import numpy as np
from mpi4py import MPI


def exchange_tensor_structure(tensor, P_send, P_recv):

    tensor_requires_grad = None
    tensor_dim = None
    tensor_sizes = None

    requests = []

    if P_send.active:

        tensor_requires_grad = tensor.requires_grad
        tensor_dim = len(tensor.shape)
        tensor_sizes = np.array(tensor.shape, dtype=np.int)

        irg_in = np.array([1 if tensor_requires_grad else 0], dtype=np.int)
        irg = np.array([-1], dtype=np.int)
        req = P_send.comm.Iallreduce(irg_in.copy(), irg, op=MPI.MAX)
        requests.append(req)

        # The output tensors do not know the size or dimension, so we must
        # share that information first.
        tensor_dim_in = np.array(len(tensor_sizes), dtype=np.int)
        tensor_dim = -1*np.ones(1, dtype=np.int)
        req = P_send.comm.Iallreduce(tensor_dim_in.copy(), tensor_dim, op=MPI.MAX)
        requests.append(req)

        tensor_sizes_in = tensor_sizes.copy()
        tensor_sizes = -1*np.ones(tensor_dim, dtype=np.int)
        req = P_send.comm.Iallreduce(tensor_sizes_in.copy(), tensor_sizes, op=MPI.MAX)
        requests.append(req)

    if P_send != P_recv and P_recv.active:
        r_irg_in = -1*np.ones(1, dtype=np.int)
        r_irg = np.array([-1], dtype=np.int)
        req = P_recv.comm.Iallreduce(r_irg_in, r_irg, op=MPI.MAX)
        req.Wait()
        r_tensor_requires_grad = bool(r_irg[0] == 1)

        r_tensor_dim_in = -1*np.ones(1, dtype=np.int)
        r_tensor_dim = -1*np.ones(1, dtype=np.int)
        req = P_recv.comm.Iallreduce(r_tensor_dim_in, r_tensor_dim, op=MPI.MAX)
        req.Wait()

        r_tensor_sizes_in = -1*np.ones(r_tensor_dim, dtype=np.int)
        r_tensor_sizes = -1*np.ones(r_tensor_dim, dtype=np.int)
        req = P_recv.comm.Iallreduce(r_tensor_sizes_in, r_tensor_sizes, op=MPI.MAX)
        req.Wait()

    MPI.Request.Waitall(requests)

    # Don't change the send buffers until they are complete!
    if P_send != P_recv and P_recv.active:
        tensor_requires_grad = r_tensor_requires_grad
        tensor_dim = r_tensor_dim
        tensor_sizes = r_tensor_sizes

    return tensor_requires_grad, tensor_dim, tensor_sizes
