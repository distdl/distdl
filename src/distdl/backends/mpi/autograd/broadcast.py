import numpy as np
import torch
from mpi4py import MPI

from distdl.utilities.torch import NoneTensor


class BroadcastFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_send, P_recv,
                input_tensor_structure, output_tensor_structure, dtype):

        ctx.P_send = P_send
        ctx.P_recv = P_recv
        ctx.input_tensor_structure = input_tensor_structure
        ctx.output_tensor_structure = output_tensor_structure
        ctx.dtype = dtype

        output_requires_grad = output_tensor_structure[0]
        out_tensor_sizes = output_tensor_structure[2]

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
        input_tensor_structure = ctx.input_tensor_structure
        output_tensor_structure = ctx.output_tensor_structure
        dtype = ctx.dtype

        input_requires_grad = input_tensor_structure[0]
        in_tensor_sizes = input_tensor_structure[2]
        out_tensor_sizes = output_tensor_structure[2]

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

        return grad_input, None, None, None, None, None
