import numpy as np
import torch
from mpi4py import MPI

from distdl.nn.exchange_tensor_structure_mixin import _ExchangeTensorStructureMixin as _exchange
from distdl.utilities.torch import NoneTensor


class BroadcastFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_bcast_same, P_bcast_send, P_bcast_recv, dtype):

        ctx.P_bcast_same = P_bcast_same
        ctx.P_bcast_send = P_bcast_send
        ctx.P_bcast_recv = P_bcast_recv
        ctx.dtype = dtype

        # From the partition creation, we guarantee that only one of these
        # will be active, if any of them are active at all.  This works
        # because if the send and recv partitions are the same, we still need
        # to send data but don't need to receive it, we will have local
        # copies.
        P_send = P_bcast_send
        if P_bcast_same.active:
            P_send = P_bcast_same
        P_recv = P_bcast_recv

        # Share the input tensor structure so the output can create space for
        # the data.
        tensor_structure = _exchange._exchange_tensor_structure(input,
                                                                P_send,
                                                                P_recv)
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

        # Send all of the data
        if P_send.active:
            input_numpy = input.detach().numpy()
            req = P_send.comm.Ibcast(input_numpy, root=0)
            requests.append(req)

        # If I both "send" and "recv" then I just make a copy of the input
        if P_bcast_same.active:
            output = input.clone()

        # If I just receive, receive the broadcast
        if P_recv.active:
            output = np.zeros(tensor_sizes, dtype=dtype)

            req = P_recv.comm.Ibcast(output, root=0)
            req.Wait()
            output = torch.tensor(output, requires_grad=input_requires_grad)

        MPI.Request.Waitall(requests)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        P_bcast_same = ctx.P_bcast_same
        P_bcast_send = ctx.P_bcast_send
        P_bcast_recv = ctx.P_bcast_recv
        dtype = ctx.dtype
        input_requires_grad = ctx.input_requires_grad
        tensor_sizes = ctx.tensor_sizes

        # Like above.
        P_send = P_bcast_send
        P_recv = P_bcast_recv
        if P_bcast_same.active:
            P_recv = P_bcast_same

        # This allows all ranks to use the same exit path, so that we can be
        # sure that all requests have cleared.
        grad_input = NoneTensor()

        requests = []

        # If I received data (either from a remote worker or just from myself)
        # I need to reduce that data.  If I I send and receive to myself, this
        # is OK, as the reduction accounts for the copy, unlike the broadcast
        # above.
        if P_recv.active:
            reduced_data = np.zeros(tensor_sizes, dtype=dtype)
            grad_output_numpy = grad_output.detach().numpy()
            req = P_recv.comm.Ireduce(grad_output_numpy, reduced_data, root=0, op=MPI.SUM)
            requests.append(req)

        # If I sent data in the forward, I have to receive it here.  mpi4py
        # does not allow aliasing of the input, so we have to make a copy of
        # nothing, unfortunately.
        if P_send.active:
            reduced_data = np.zeros(tensor_sizes, dtype=dtype)
            req = P_bcast_send.comm.Ireduce(reduced_data.copy(), reduced_data, root=0, op=MPI.SUM)
            requests.append(req)

        MPI.Request.Waitall(requests)

        # If we had to receive data, we need to tensorify it.
        if P_send.active or P_bcast_same.active:
            grad_input = torch.tensor(reduced_data, requires_grad=input_requires_grad)

        return grad_input, None, None, None, None, None


class Broadcast(torch.nn.Module):

    def __init__(self, P_in, P_out):
        super(Broadcast, self).__init__()

        self.P_in = P_in
        self.P_out = P_out

        # TODO: #25  Make selection of dtype more sensible.
        self.dtype = np.float32

        if P_in == P_out:
            self.identity = True
        else:
            self.identity = False
            bcast_partitions = P_in.create_broadcast_partition_to(P_out)
            self.P_bcast_same = bcast_partitions[0]
            self.P_bcast_send = bcast_partitions[1]
            self.P_bcast_recv = bcast_partitions[2]

    def forward(self, input):

        if self.identity:
            return input.clone()

        return BroadcastFunction.apply(input,
                                       self.P_bcast_same,
                                       self.P_bcast_send,
                                       self.P_bcast_recv,
                                       self.dtype)
