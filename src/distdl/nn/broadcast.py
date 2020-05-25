import numpy as np
import torch
from mpi4py import MPI

from distdl.nn.exchange_tensor_structure_mixin import _ExchangeTensorStructureMixin


class BroadcastFunction(torch.autograd.Function,
                        _ExchangeTensorStructureMixin):

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

        tensor_structure = ctx._exchange_tensor_structure(input,
                                                          P_send,
                                                          P_recv)
        input_requires_grad = tensor_structure[0]
        tensor_dim = tensor_structure[1]
        tensor_sizes = tensor_structure[2]

        ctx.input_requires_grad = input_requires_grad
        ctx.tensor_dim = tensor_dim
        ctx.tensor_sizes = tensor_sizes

        output = None

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

            req = P_bcast_recv.comm.Ibcast(output, root=0)
            req.Wait()
            output = torch.tensor(output, requires_grad=input_requires_grad)

        MPI.Request.Waitall(requests)

        # # If I am both a source and destination in the same partition, I need
        # # to share my input with everyone, but also make my own output copy.
        # if P_bcast_same.active:
        #     input_numpy = input.detach().numpy()
        #     req = P_bcast_same.comm.Ibcast(input_numpy, root=0)
        #     requests.append(req)
        #     output = input.clone()
        # # Otherwise, I either only share my data or I recieve my output
        # else:
        #     if P_bcast_send.active:
        #         input_numpy = input.detach().numpy()
        #         req = P_bcast_send.comm.Ibcast(input_numpy, root=0)
        #         requests.append(req)
        #     if P_bcast_recv.active:
        #         output = np.zeros(tensor_sizes, dtype=dtype)
        #         req = P_bcast_recv.comm.Ibcast(output, root=0)
        #         req.Wait()
        #         output = torch.tensor(output,
        #                               requires_grad=input_requires_grad)

        # # Ensure that all send requests have cleared.
        # MPI.Request.Waitall(requests)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        P_common = ctx.P_common
        P_in = ctx.P_in
        P_out = ctx.P_out
        in_root = ctx.in_root
        out_root = ctx.out_root
        dtype = ctx.dtype

        # We need to move data between two potentially disjoint partitions,
        # so we have to find a common mapping, which requires the common
        # partition.
        P_common_to_P_in = P_in.map_from_ancestor(P_common)
        P_common_to_P_out = P_out.map_from_ancestor(P_common)

        # Collect at least the send requests so they can resolve out of order.
        requests = []

        # The output partition performs the adjoint of the broadcast, a sum
        # reduction.  The root rank of the output partition then sends that
        # to the root rank of the input partition, completing the operation.
        if P_out.active:
            grad_output_numpy = grad_output.detach().numpy()
            reduced_data = np.zeros(shape=grad_output_numpy.shape,
                                    dtype=grad_output_numpy.dtype)
            P_out.comm.Reduce(grad_output_numpy, reduced_data,
                              root=out_root, op=MPI.SUM)

            if P_out.rank == out_root:
                partner = np.where(P_common_to_P_in == in_root)[0][0]
                req = P_common.comm.Isend(reduced_data, dest=partner, tag=1235)
                requests.append(req)

        # This allows all ranks to use the same exit path, so that we can be
        # sure that all requests have cleared.
        grad_input = None

        # Only the root rank of the input partition receives the data and
        # returns the result.
        if P_in.active and P_in.rank == in_root:
            tensor_sizes = ctx.tensor_sizes
            input_requires_grad = ctx.input_requires_grad

            grad_input = np.zeros(tensor_sizes, dtype=dtype)

            partner = np.where(P_common_to_P_out == out_root)[0][0]
            req = P_common.comm.Irecv(grad_input, source=partner, tag=1235)
            req.Wait()

            grad_input = torch.tensor(grad_input, requires_grad=input_requires_grad)

        # Ensure all sends have finished.
        MPI.Request.Waitall(requests)

        return grad_input, None, None, None, None, None


class Broadcast(torch.nn.Module):

    def __init__(self, P_in, P_out):
        super(Broadcast, self).__init__()

        self.P_in = P_in
        self.P_out = P_out

        bcast_partitions = P_in.create_broadcast_partition_to(P_out)
        self.P_bcast_same = bcast_partitions[0]
        self.P_bcast_send = bcast_partitions[1]
        self.P_bcast_recv = bcast_partitions[2]

        # TODO: #25  Make selection of dtype more sensible.
        self.dtype = np.float32

    def forward(self, input):

        # If we are not sending or receving any data
        if (not self.P_bcast_same.active and
            not self.P_bcast_send.active and
            not self.P_bcast_recv.active): # noqa E129
            return None

        return BroadcastFunction.apply(input,
                                       self.P_bcast_same,
                                       self.P_bcast_send,
                                       self.P_bcast_recv,
                                       self.dtype)
