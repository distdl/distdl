__all__ = ["HaloExchangeFunction"]

import numpy as np
import torch
from mpi4py import MPI

from distdl.utilities.torch import zero_volume_tensor


class HaloExchangeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_x, slices, buffers, neighbor_ranks):

        ctx.slices = slices
        ctx.buffers = buffers
        ctx.neighbor_ranks = neighbor_ranks
        ctx.P_x = P_x

        if not P_x.active:
            return zero_volume_tensor(input.shape[0])

        ctx.mark_dirty(input)

        if P_x.size == 1:
            return input

        input_numpy = input.detach().numpy()

        dim = P_x.dim
        for i in range(dim):

            lbs, lgs, rbs, rgs = slices[i]
            lbb, lgb, rbb, rgb = buffers[i]
            lrank, rrank = neighbor_ranks[i]

            if lbb is not None:
                np.copyto(lbb, input_numpy[lbs].ravel())
            if rbb is not None:
                np.copyto(rbb, input_numpy[rbs].ravel())

            ltag = 0
            rtag = 1

            lrecv_req = P_x._comm.Irecv(lgb, source=lrank, tag=rtag) if lgb is not None else MPI.REQUEST_NULL
            rrecv_req = P_x._comm.Irecv(rgb, source=rrank, tag=ltag) if rgb is not None else MPI.REQUEST_NULL
            lsend_req = P_x._comm.Isend(lbb, dest=lrank, tag=ltag) if lbb is not None else MPI.REQUEST_NULL
            rsend_req = P_x._comm.Isend(rbb, dest=rrank, tag=rtag) if rbb is not None else MPI.REQUEST_NULL

            reqs = [lrecv_req, rrecv_req, lsend_req, rsend_req]
            n_reqs_completed = 0

            while n_reqs_completed < len(reqs):
                status = MPI.Status()
                index = MPI.Request.Waitany(reqs, status)

                if index != MPI.UNDEFINED:
                    if index == 0:
                        newshape = input_numpy[lgs].shape
                        np.copyto(input_numpy[lgs], lgb.reshape(newshape))
                    elif index == 1:
                        newshape = input_numpy[rgs].shape
                        np.copyto(input_numpy[rgs], rgb.reshape(newshape))

                n_reqs_completed += 1

        return input

    @staticmethod
    def backward(ctx, grad_output):

        slices = ctx.slices
        buffers = ctx.buffers
        neighbor_ranks = ctx.neighbor_ranks
        P_x = ctx.P_x

        if not P_x.active:
            return zero_volume_tensor(grad_output.shape[0]), None, None, None, None

        if P_x.size == 1:
            return grad_output, None, None, None, None

        grad_output_numpy = grad_output.detach().numpy()

        dim = P_x.dim
        for i in reversed(range(dim)):

            lbs, lgs, rbs, rgs = slices[i]
            lbb, lgb, rbb, rgb = buffers[i]
            lrank, rrank = neighbor_ranks[i]

            if lgb is not None:
                np.copyto(lgb, grad_output_numpy[lgs].ravel())
                grad_output_numpy[lgs] = 0.0
            if rgb is not None:
                np.copyto(rgb, grad_output_numpy[rgs].ravel())
                grad_output_numpy[rgs] = 0.0

            ltag = 0
            rtag = 1

            lrecv_req = P_x._comm.Irecv(lbb, source=lrank, tag=rtag) if lbb is not None else MPI.REQUEST_NULL
            rrecv_req = P_x._comm.Irecv(rbb, source=rrank, tag=ltag) if rbb is not None else MPI.REQUEST_NULL
            lsend_req = P_x._comm.Isend(lgb, dest=lrank, tag=ltag) if lgb is not None else MPI.REQUEST_NULL
            rsend_req = P_x._comm.Isend(rgb, dest=rrank, tag=rtag) if rgb is not None else MPI.REQUEST_NULL

            reqs = [lrecv_req, rrecv_req, lsend_req, rsend_req]
            n_reqs_completed = 0

            while n_reqs_completed < len(reqs):
                status = MPI.Status()
                index = MPI.Request.Waitany(reqs, status)

                if index != MPI.UNDEFINED:
                    if index == 0:
                        newshape = grad_output_numpy[lbs].shape
                        grad_output_numpy[lbs] += lbb.reshape(newshape)
                    elif index == 1:
                        newshape = grad_output_numpy[rbs].shape
                        grad_output_numpy[rbs] += rbb.reshape(newshape)

                n_reqs_completed += 1

        return grad_output, None, None, None, None
