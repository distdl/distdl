__all__ = ["HaloExchangeFunction"]

import numpy as np
import torch
from mpi4py import MPI

from distdl.utilities.slicing import compute_nd_slice_shape
from distdl.utilities.torch import zero_volume_tensor


class HaloExchangeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_x, slices, buffers, neighbor_ranks):

        device = input.device
        ctx.slices = slices
        ctx.buffers = buffers
        ctx.neighbor_ranks = neighbor_ranks
        ctx.P_x = P_x
        ctx.device = device

        if not P_x.active:
            return zero_volume_tensor(input.shape[0], device=device)

        ctx.mark_dirty(input)

        if P_x.size == 1:
            return input

        dim = P_x.dim
        for i in range(dim):

            lbs, lgs, rbs, rgs = slices[i]
            lbb, lgb, rbb, rgb = buffers[i]
            if lbb is not None:
                lbb = lbb.get_view(compute_nd_slice_shape(lbs))
            if lgb is not None:
                lgb = lgb.get_view(compute_nd_slice_shape(lgs))
            if rbb is not None:
                rbb = rbb.get_view(compute_nd_slice_shape(rbs))
            if rgb is not None:
                rgb = rgb.get_view(compute_nd_slice_shape(rgs))
            lrank, rrank = neighbor_ranks[i]

            if lbb is not None:
                np.copyto(lbb, input.detach()[lbs].cpu().numpy())
            if rbb is not None:
                np.copyto(rbb, input.detach()[rbs].cpu().numpy())

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
                        input[lgs] = torch.tensor(lgb, device=device)
                    elif index == 1:
                        input[rgs] = torch.tensor(rgb, device=device)

                n_reqs_completed += 1

        return input

    @staticmethod
    def backward(ctx, grad_output):

        slices = ctx.slices
        buffers = ctx.buffers
        neighbor_ranks = ctx.neighbor_ranks
        P_x = ctx.P_x
        device = ctx.device

        assert grad_output.device == device

        if not P_x.active:
            return zero_volume_tensor(grad_output.shape[0], device=device), None, None, None, None

        if P_x.size == 1:
            return grad_output, None, None, None, None

        ctx.mark_dirty(grad_output)

        dim = P_x.dim
        for i in reversed(range(dim)):

            lbs, lgs, rbs, rgs = slices[i]
            lbb, lgb, rbb, rgb = buffers[i]
            if lbb is not None:
                lbb = lbb.get_view(compute_nd_slice_shape(lbs))
            if lgb is not None:
                lgb = lgb.get_view(compute_nd_slice_shape(lgs))
            if rbb is not None:
                rbb = rbb.get_view(compute_nd_slice_shape(rbs))
            if rgb is not None:
                rgb = rgb.get_view(compute_nd_slice_shape(rgs))
            lrank, rrank = neighbor_ranks[i]

            if lgb is not None:
                np.copyto(lgb, grad_output.detach()[lgs].cpu().numpy())
                grad_output[lgs] = 0.0
            if rgb is not None:
                np.copyto(rgb, grad_output.detach()[rgs].cpu().numpy())
                grad_output[rgs] = 0.0

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
                        grad_output[lbs] += torch.tensor(lbb, device=device)
                    elif index == 1:
                        grad_output[rbs] += torch.tensor(rbb, device=device)

                n_reqs_completed += 1

        return grad_output, None, None, None, None
