import numpy as np
from mpi4py import MPI


# Holy cow this is a touchy function, be very careful if modifying it...
def exchange_tensor_structure(tensor, P_send, P_recv):

    if not P_send.active and not P_recv.active:
        return None, None, None

    requests = []

    # Pre-allocate as many communication variables as possible.
    rg_int = np.array([-10000], dtype=np.int)
    rg_int_red = np.array([0], dtype=np.int)

    tensor_dim_in = np.zeros(1, np.int)
    tensor_dim_out = np.zeros(1, np.int)

    # Cannot pre-allocate these.  Many ranks need the former values first.
    tensor_sizes_in = None
    tensor_sizes_out = None

    if P_send.active:

        # Need to send non-Python types, so convert the boolean temporarily
        rg_int[0] = 1 if tensor.requires_grad else 0
        req = P_send.comm.Iallreduce(rg_int, rg_int_red, op=MPI.MAX)
        requests.append(req)

        # Sending processes know the shape, so they can set the send data, but
        # should still use null receive data
        tensor_dim_in[0] = len(tensor.shape)
        req = P_send.comm.Iallreduce(tensor_dim_in, tensor_dim_out, op=MPI.MAX)
        requests.append(req)

        # Similarly, sending processes know the tensor shape, but we allocate
        # space here for symmetry and to ensure that only processes that know
        # the data try to allocate something.
        tensor_sizes_in = np.array(tensor.shape, dtype=np.int)
        tensor_sizes_out = np.zeros(len(tensor.shape), dtype=np.int)
        req = P_send.comm.Iallreduce(tensor_sizes_in, tensor_sizes_out, op=MPI.MAX)
        requests.append(req)

    # If the process is a receiving process, but doesn't already know the data
    # because it is the _same_ sending process, then we receive the results.
    if (P_send != P_recv) and P_recv.active:

        # Everyone needs to receive this, but we don't need it for future
        # communication so we can defer receiving the data.
        req = P_recv.comm.Iallreduce(rg_int, rg_int_red, op=MPI.MAX)
        requests.append(req)

        # We need this value for the next communication, so we have to wait
        # for it to complete before moving on.
        tensor_dim_in[0] = -1
        req = P_recv.comm.Iallreduce(tensor_dim_in, tensor_dim_out, op=MPI.MAX)
        req.Wait()

        tensor_sizes_in = np.zeros(tensor_dim_out, dtype=np.int)-100000
        tensor_sizes_out = np.zeros(tensor_dim_out, dtype=np.int)
        req = P_recv.comm.Iallreduce(tensor_sizes_in, tensor_sizes_out, op=MPI.MAX)
        requests.append(req)

    MPI.Request.Waitall(requests)

    tensor_requires_grad = bool(rg_int_red[0] == 1)
    tensor_dim = tensor_dim_out
    tensor_sizes = tensor_sizes_out

    return tensor_requires_grad, tensor_dim, tensor_sizes
