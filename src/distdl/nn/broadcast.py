import numpy as np
import torch
from mpi4py import MPI

from distdl.backends.mpi.exchange_tensor import exchange_tensor_structure
from distdl.nn.module import Module
from distdl.utilities.torch import NoneTensor


class BroadcastFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_send, P_recv, dtype):

        ctx.P_send = P_send
        ctx.P_recv = P_recv
        ctx.dtype = dtype

        input_requires_grad = input.requires_grad
        in_tensor_dim = len(input.shape)
        in_tensor_sizes = np.array(input.shape, dtype=np.int)
        # Share the input tensor structure so the output can create space for
        # the data.
        out_tensor_structure = exchange_tensor_structure(input, P_send, P_recv)
        output_requires_grad = out_tensor_structure[0]
        out_tensor_dim = out_tensor_structure[1]
        out_tensor_sizes = out_tensor_structure[2]

        ctx.input_requires_grad = input_requires_grad
        ctx.in_tensor_dim = in_tensor_dim
        ctx.in_tensor_sizes = in_tensor_sizes
        ctx.output_requires_grad = output_requires_grad
        ctx.out_tensor_dim = out_tensor_dim
        ctx.out_tensor_sizes = out_tensor_sizes

        # This allows all ranks to use the same exit path, so that we can be
        # sure that all requests have cleared.
        output = NoneTensor()

        # return output
        requests = []

        # Send all of the data
        if P_send.active:
            input_numpy = input.detach().numpy()
            req = P_send.comm.Ibcast(input_numpy, root=0)
            requests.append(req)

        if P_recv.active:
            # If I also send, make a copy.
            if P_send == P_recv:
                output = input.clone()
            # If I just receive, receive the broadcast
            else:
                output = np.zeros(out_tensor_sizes, dtype=dtype)

                req = P_recv.comm.Ibcast(output, root=0)
                req.Wait()
                output = torch.tensor(output, requires_grad=output_requires_grad)

        MPI.Request.Waitall(requests)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        P_send = ctx.P_send
        P_recv = ctx.P_recv
        dtype = ctx.dtype
        input_requires_grad = ctx.input_requires_grad
        in_tensor_sizes = ctx.in_tensor_sizes
        out_tensor_sizes = ctx.out_tensor_sizes

        # This allows all ranks to use the same exit path, so that we can be
        # sure that all requests have cleared.
        grad_input = NoneTensor()

        requests = []

        # If I received data (either from a remote worker or just from myself)
        # I need to reduce that data.  If I send and receive to myself, this
        # is OK, as the reduction accounts for the copy, unlike the broadcast
        # above.
        if P_recv.active:
            reduced_data_recv = np.zeros(out_tensor_sizes, dtype=dtype)
            grad_output_numpy = grad_output.detach().numpy()
            req = P_recv.comm.Ireduce(grad_output_numpy, reduced_data_recv, root=0, op=MPI.SUM)
            requests.append(req)

        # If I sent data in the forward, I have to receive it here.  Unless I
        # also received that data, then I already have it from abive.  mpi4py
        # does not allow aliasing of the input, so we have to make a copy of
        # nothing, unfortunately.
        if P_send != P_recv and P_send.active:
            reduced_data_send = np.zeros(in_tensor_sizes, dtype=dtype)
            req = P_send.comm.Ireduce(reduced_data_send.copy(), reduced_data_send, root=0, op=MPI.SUM)
            requests.append(req)

        MPI.Request.Waitall(requests)

        # If we had to receive data, we need to tensorify it.
        if P_send.active:
            if P_send == P_recv:
                grad_input = torch.tensor(reduced_data_recv, requires_grad=input_requires_grad)
            else:
                grad_input = torch.tensor(reduced_data_send, requires_grad=input_requires_grad)

        return grad_input, None, None, None


class Broadcast(Module):

    def __init__(self, P_in, P_out, transpose_src=False, transpose_dest=False):
        super(Broadcast, self).__init__()

        self.P_in = P_in
        self.P_out = P_out

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
            bcast_partitions = P_in.create_broadcast_partition_to(P_out,
                                                                  transpose_src,
                                                                  transpose_dest)
            self.P_send = bcast_partitions[0]
            self.P_recv = bcast_partitions[1]

    def forward(self, input):

        if self.identity:
            return input.clone()

        return BroadcastFunction.apply(input,
                                       self.P_send,
                                       self.P_recv,
                                       self.dtype)
