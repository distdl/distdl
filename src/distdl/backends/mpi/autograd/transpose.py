import numpy as np
import torch
from mpi4py import MPI

from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import NoneTensor


class DistributedTransposeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_union, x_global_shape,
                P_x, in_data, in_buffers,
                P_y, out_data, out_buffers, dtype):

        ctx.P_union = P_union
        ctx.x_global_shape = x_global_shape

        ctx.P_x = P_x
        ctx.in_data = in_data
        ctx.in_buffers = in_buffers

        ctx.P_y = P_y
        ctx.out_data = out_data
        ctx.out_buffers = out_buffers

        ctx.dtype = dtype

        input_requires_grad = False
        # By design, P_x is always first in the union
        if P_union.active:
            if P_x.rank == 0:
                input_requires_grad = input.requires_grad
                P_union.comm.Bcast(np.array([1 if input_requires_grad else 0]),
                                   root=0)
            else:
                irg = np.array([0], dtype=np.int)
                P_union.comm.Bcast(irg, root=0)
                input_requires_grad = bool(irg[0] == 1)

        ctx.input_requires_grad = input_requires_grad

        requests = []

        # Default everyone to output nothing
        output = NoneTensor()

        # If I am getting data, recv my output parts
        recv_count = 0
        if P_y.active:
            for (sl, sz, partner), buff in zip(out_data, out_buffers):
                if buff is not None:
                    req = P_union.comm.Irecv(buff, source=partner, tag=111)
                    requests.append(req)
                else:
                    # We add this if there is no recv so that the indices of
                    # the requests array match the indices of out_data and
                    # out_buffers.
                    requests.append(MPI.REQUEST_NULL)
                recv_count += 1

        # If I have data to share, pack and send my input parts
        send_count = 0
        if P_x.active:
            input_numpy = input.detach().numpy()
            for (sl, sz, partner), buff in zip(in_data, in_buffers):
                if buff is not None:
                    np.copyto(buff, input_numpy[tuple(sl)].ravel())
                    req = P_union.comm.Isend(buff, dest=partner, tag=111)
                    requests.append(req)
                else:
                    # We add this for symmetry, but don't really need it.
                    requests.append(MPI.REQUEST_NULL)
                send_count += 1

        # We do this after the sends so that they can get started before local
        # allocations.
        if P_y.active:
            index = P_y.index
            y_local_shape = compute_subshape(P_y.shape, index, x_global_shape)
            # TODO(#25): The dtype should not be fixed, but correcting this is
            #            a thing that needs to be resolved globally.
            output = np.zeros(y_local_shape, dtype=dtype)

        # Unpack the received data as it arrives
        completed_count = 0
        while(completed_count < len(requests)):
            status = MPI.Status()
            index = MPI.Request.Waitany(requests, status)

            # In MPI, we don't get the index out if the request is an
            # instance of MPI.REQUEST_NULL, instead MPI.UNDEFINED is returned.
            if P_y.active and index < recv_count and index != MPI.UNDEFINED:
                # Unpack my output parts
                sl, sz, partner = out_data[index]
                buff = out_buffers[index]
                if buff is not None:
                    sh = output[tuple(sl)].shape
                    np.copyto(output[tuple(sl)], buff.reshape(sh))

            completed_count += 1

        if P_y.active:
            output = torch.from_numpy(output)
            output.requires_grad = input_requires_grad

        return output

    @staticmethod
    def backward(ctx, grad_output):

        P_union = ctx.P_union
        x_global_shape = ctx.x_global_shape

        P_x = ctx.P_x
        in_data = ctx.in_data
        in_buffers = ctx.in_buffers

        P_y = ctx.P_y
        out_data = ctx.out_data
        out_buffers = ctx.out_buffers

        dtype = ctx.dtype

        input_requires_grad = ctx.input_requires_grad

        requests = []

        # Default everyone to output None
        grad_input = NoneTensor()

        # Recv my input parts
        recv_count = 0
        if P_x.active:
            for (sl, sz, partner), buff in zip(in_data, in_buffers):
                if buff is not None:
                    req = P_union.comm.Irecv(buff, source=partner, tag=113)
                    requests.append(req)
                else:
                    # We add this if there is no recv so that the indices of
                    # the requests array match the indices of in_data and
                    # in_buffers.
                    requests.append(MPI.REQUEST_NULL)
                recv_count += 1

        # Pack and send my input parts
        send_count = 0
        if P_y.active:
            grad_output_numpy = grad_output.detach().numpy()
            for (sl, sz, partner), buff in zip(out_data, out_buffers):
                if buff is not None:
                    np.copyto(buff, grad_output_numpy[tuple(sl)].ravel())
                    req = P_union.comm.Isend(buff, dest=partner, tag=113)
                    requests.append(req)
                else:
                    # We add this for symmetry, but don't really need it.
                    requests.append(MPI.REQUEST_NULL)
                send_count += 1

        if P_x.active:
            index = P_x.index
            x_local_shape = compute_subshape(P_x.shape, index, x_global_shape)
            # TODO(#25): The dtype should not be fixed, but correcting this is
            #            a thing that needs to be resolved globally.
            grad_input = np.zeros(x_local_shape, dtype=dtype)

        # Unpack the received data as it arrives
        completed_count = 0
        while(completed_count < len(requests)):
            status = MPI.Status()
            index = MPI.Request.Waitany(requests, status)

            # In MPI, we don't get the index out if the request is an
            # instance of MPI.REQUEST_NULL, instead MPI.UNDEFINED is returned.
            if P_x.active and index < recv_count and index != MPI.UNDEFINED:
                # Unpack my output parts
                sl, sz, partner = in_data[index]
                buff = in_buffers[index]
                if buff is not None:
                    sh = grad_input[tuple(sl)].shape
                    # This would normally be an add into the grad_input tensor
                    # but we just created it, so a copy is sufficient.
                    np.copyto(grad_input[tuple(sl)], buff.reshape(sh))

            completed_count += 1

        if P_x.active:
            grad_input = torch.from_numpy(grad_input)
            grad_input.requires_grad = input_requires_grad

        return grad_input, None, None, None, None, None, None, None, None, None
