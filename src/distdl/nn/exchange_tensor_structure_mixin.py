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
            irg = np.array([1 if tensor_requires_grad else 0], dtype=np.int)
            req = P_send.comm.Ibcast(irg, root=0)
            requests.append(req)

            # The output tensors do not know the size or dimension, so we must
            # share that information first.
            tensor_dim = np.array(len(tensor_numpy.shape), dtype=np.int)
            req = P_send.comm.Ibcast(tensor_dim, root=0)
            requests.append(req)

            tensor_sizes = np.array(tensor_numpy.shape, dtype=np.int)
            req = P_send.comm.Ibcast(tensor_sizes, root=0)
            requests.append(req)

        if P_recv.active:
            # mpi4py does not support lowercase ibcast, so we have to hack it
            irg = np.zeros(1, dtype=np.int) - 1
            req = P_recv.comm.Ibcast(irg, root=0)
            req.Wait()
            tensor_requires_grad = bool(irg[0] == 1)

            tensor_dim = np.zeros(1, dtype=np.int)
            req = P_recv.comm.Ibcast(tensor_dim, root=0)
            req.Wait()

            tensor_sizes = np.zeros(tensor_dim, dtype=np.int)
            req = P_recv.comm.Ibcast(tensor_sizes, root=0)
            req.Wait()

        MPI.Request.Waitall(requests)

        return tensor_requires_grad, tensor_dim, tensor_sizes
