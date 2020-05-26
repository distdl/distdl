import numpy as np
from mpi4py import MPI


class _ExchangeTensorStructureMixin:

    @classmethod
    def _exchange_tensor_structure(cls, tensor, P_send, P_recv):

        tensor_requires_grad = None
        tensor_dim = None
        tensor_sizes = None

        requests = []

        if P_send.active:

            tensor_numpy = tensor.detach().numpy()

            tensor_requires_grad = tensor.requires_grad
            # mpi4py does not support lowercase ibcast, so we have to hack it
            irg_in = np.array([1 if tensor_requires_grad else 0], dtype=np.int)
            irg = np.array([-1], dtype=np.int)
            req = P_send.comm.Iallreduce(irg_in, irg, op=MPI.MAX)
            requests.append(req)

            # The output tensors do not know the size or dimension, so we must
            # share that information first.
            tensor_dim_in = np.array(len(tensor_numpy.shape), dtype=np.int)
            tensor_dim = -1*np.ones(1, dtype=np.int)
            req = P_send.comm.Iallreduce(tensor_dim_in, tensor_dim, op=MPI.MAX)
            requests.append(req)

            tensor_sizes_in = np.array(tensor_numpy.shape, dtype=np.int)
            tensor_sizes = -1*np.ones(len(tensor_numpy.shape), dtype=np.int)
            req = P_send.comm.Iallreduce(tensor_sizes_in, tensor_sizes, op=MPI.MAX)
            requests.append(req)

        if P_recv.active:
            # mpi4py does not support lowercase ibcast, so we have to hack it
            irg_in = -1*np.ones(1, dtype=np.int)
            irg = np.array([-1], dtype=np.int)
            req = P_recv.comm.Iallreduce(irg_in, irg, op=MPI.MAX)
            req.Wait()
            tensor_requires_grad = bool(irg[0] == 1)

            tensor_dim_in = -1*np.ones(1, dtype=np.int)
            tensor_dim = -1*np.ones(1, dtype=np.int)
            req = P_recv.comm.Iallreduce(tensor_dim_in, tensor_dim, op=MPI.MAX)
            req.Wait()

            tensor_sizes_in = -1*np.ones(tensor_dim, dtype=np.int)
            tensor_sizes = -1*np.ones(tensor_dim, dtype=np.int)
            req = P_recv.comm.Iallreduce(tensor_sizes_in, tensor_sizes, op=MPI.MAX)
            req.Wait()

        MPI.Request.Waitall(requests)

        return tensor_requires_grad, tensor_dim, tensor_sizes
