import numpy as np
import torch
from mpi4py import MPI

from distdl.utilities.torch import NoneTensor


class SumReduceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_send, P_recv,
                input_tensor_structure, output_tensor_structure, dtype):

        ctx.P_send = P_send
        ctx.P_recv = P_recv
        ctx.input_tensor_structure = input_tensor_structure
        ctx.output_tensor_structure = output_tensor_structure
        ctx.dtype = dtype

        in_tensor_sizes = input_tensor_structure[2]
        output_requires_grad = output_tensor_structure[0]
        out_tensor_sizes = output_tensor_structure[2]

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
            reduced_data_send = np.zeros(in_tensor_sizes, dtype=dtype)
            input_numpy = input.detach().numpy()
            req = P_send.comm.Ireduce(input_numpy, reduced_data_send, root=0, op=MPI.SUM)
            requests.append(req)

        # If I sent data in the forward, I have to receive it here.  mpi4py
        # does not allow aliasing of the input, so we have to make a copy of
        # nothing, unfortunately.
        if P_send != P_recv and P_recv.active:
            reduced_data_recv = np.zeros(out_tensor_sizes, dtype=dtype)
            req = P_recv.comm.Ireduce(reduced_data_recv.copy(), reduced_data_recv, root=0, op=MPI.SUM)
            requests.append(req)

        MPI.Request.Waitall(requests)

        # If we had to receive data, we need to tensorify it.
        if P_recv.active:
            if P_send == P_recv:
                output = torch.tensor(reduced_data_send, requires_grad=output_requires_grad)
            else:
                output = torch.tensor(reduced_data_recv, requires_grad=output_requires_grad)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        P_send = ctx.P_send
        P_recv = ctx.P_recv
        input_tensor_structure = ctx.input_tensor_structure
        dtype = ctx.dtype

        input_requires_grad = input_tensor_structure[0]
        in_tensor_sizes = input_tensor_structure[2]

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
                grad_input = np.zeros(in_tensor_sizes, dtype=dtype)

                req = P_send.comm.Ibcast(grad_input, root=0)
                req.Wait()
                grad_input = torch.tensor(grad_input,
                                          requires_grad=input_requires_grad)

        MPI.Request.Waitall(requests)

        return grad_input, None, None, None, None, None
