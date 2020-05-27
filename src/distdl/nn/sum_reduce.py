import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.exchange_tensor import exchange_tensor_structure
from distdl.utilities.torch import NoneTensor


class SumReduceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_send, P_recv, dtype):

        ctx.P_send = P_send
        ctx.P_recv = P_recv
        ctx.dtype = dtype

        # Share the input tensor structure so the output can create space for
        # the data.
        tensor_structure = exchange_tensor_structure(input, P_send, P_recv)
        input_requires_grad = tensor_structure[0]
        tensor_dim = tensor_structure[1]
        tensor_sizes = tensor_structure[2]

        ctx.input_requires_grad = input_requires_grad
        ctx.tensor_dim = tensor_dim
        ctx.tensor_sizes = tensor_sizes

        # This allows all ranks to use the same exit path, so that we can be
        # sure that all requests have cleared.
        output = NoneTensor()

        requests = []

        # By design, the roots are always 0 in the cross-communicators
        # If I receive data (either from a remote worker or just from myself)
        # I need to reduce that data.  If I send and receive to myself, this
        # is OK, as the reduction accounts for the copy, unlike the broadcast
        # below.
        if P_send.active:
            reduced_data = np.zeros(tensor_sizes, dtype=dtype)
            input_numpy = input.detach().numpy()
            req = P_send.comm.Ireduce(input_numpy, reduced_data, root=0, op=MPI.SUM)
            requests.append(req)

        # If I sent data in the forward, I have to receive it here.  mpi4py
        # does not allow aliasing of the input, so we have to make a copy of
        # nothing, unfortunately.
        if P_send != P_recv and P_recv.active:
            reduced_data = np.zeros(tensor_sizes, dtype=dtype)
            req = P_recv.comm.Ireduce(reduced_data.copy(), reduced_data, root=0, op=MPI.SUM)
            requests.append(req)

        MPI.Request.Waitall(requests)

        # If we had to receive data, we need to tensorify it.
        if P_recv.active:
            output = torch.tensor(reduced_data, requires_grad=input_requires_grad)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        P_send = ctx.P_send
        P_recv = ctx.P_recv
        dtype = ctx.dtype
        input_requires_grad = ctx.input_requires_grad
        tensor_sizes = ctx.tensor_sizes

        # This allows all ranks to use the same exit path, so that we can be
        # sure that all requests have cleared.
        grad_input = NoneTensor()

        requests = []

        # If I received the reduction in the forward call, I broadcast my data
        if P_recv.active:
            grad_output_numpy = grad_output.detach().numpy()
            req = P_recv.comm.Ibcast(grad_output_numpy, root=0)
            requests.append(req)

        # If I just receive, receive the broadcast
        if P_send.active:
            # If I both sent and received reduction data, then I copy the "input"
            if P_send == P_recv:
                grad_input = grad_output.clone()
            else:
                grad_input = np.zeros(tensor_sizes, dtype=dtype)

                req = P_send.comm.Ibcast(grad_input, root=0)
                req.Wait()
                grad_input = torch.tensor(grad_input,
                                          requires_grad=input_requires_grad)

        MPI.Request.Waitall(requests)

        return grad_input, None, None, None


class SumReduce(torch.nn.Module):

    def __init__(self, P_in, P_out, transpose_src=False, transpose_dest=False):
        super(SumReduce, self).__init__()

        self.P_in = P_in
        self.P_out = P_out

        self.transpose_src = transpose_src
        self.transpose_dest = transpose_dest

        # TODO: #25  Make selection of dtype more sensible.
        self.dtype = np.float32

        self.identity = False

        # The identity case is if the partitions are of size 1,
        # or they are the same partition and neither is tranposed,
        # or they are the same partition and both are transposed.
        if P_in == P_out:
            if P_in.size == 1:
                self.identity = True
            elif (transpose_dest and transpose_src) or \
                 (not transpose_dest and not transpose_src):
                self.identity = True

        # We do the actual work if it is not an identity
        if not self.identity:
            reduce_partitions = P_in.create_reduction_partition_to(P_out,
                                                                   transpose_src,
                                                                   transpose_dest)
            self.P_send = reduce_partitions[0]
            self.P_recv = reduce_partitions[1]

    def forward(self, input):

        if self.identity:
            return input.clone()

        return SumReduceFunction.apply(input,
                                       self.P_send,
                                       self.P_recv,
                                       self.dtype)
