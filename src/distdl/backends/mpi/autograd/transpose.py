__all__ = ["DistributedTransposeFunction"]

import numpy as np
import torch
from mpi4py import MPI

from distdl.utilities.dtype import torch_to_numpy_dtype_dict
from distdl.utilities.torch import zero_volume_tensor


class DistributedTransposeFunction(torch.autograd.Function):
    r"""MPI-based functional implementation of a distributed transpose layer.

    Implements the required `forward()` and adjoint (`backward()`) operations
    for a distributed Transpose layer using the PyTorch autograd interface.

    This implementation uses MPI for data movement, accessed through the
    ``mpi4py`` MPI wrappers.

    Warning
    -------
    This implementation currently requires that tensors have data stored in main
    memory (CPU) only, not auxiliary memories such as those on GPUs.

    Warning
    -------
    The ``mpi4py`` interface currently used requires NumPy views of the tensors.

    """

    @staticmethod
    def forward(ctx, input, P_union, x_global_structure,
                x_local_structure, y_local_structure,
                P_x, P_x_to_y_overlaps, P_x_to_y_buffers,
                P_y, P_y_to_x_overlaps, P_y_to_x_buffers, preserve_batch):
        r"""Forward function of distributed transpose layer.

        This method implements the forward transpose operation using MPI
        immediate-mode, non-blocking communication.

        Any given worker may send data to multiple workers in ``P_y`` and
        receive data from multiple workers in ``P_x``.  All communication
        across partitions occurs through the ``P_union`` partition.

        Data is copied using ``MPI_Irecv`` and ``MPI_Isend``. As is standard
        procedure, the receives are posted first, allowing them to complete as
        they can.  Then, buffers are packed and sent.  Once all sends have
        been posted, received data is unpacked in the order that the receives
        complete.

        When the current worker is inactive in the ``P_y`` partition, it will
        output a zero-volume tensor, potentially preserving a non-zero batch
        size.

        Parameters
        ----------
        ctx :
            PyTorch context.
        input : `torch.tensor`
            Input tensor.
        P_union : Partition
            Partition through which all communication occurs.
        x_global_structure :
            Structure of the global input tensor.
        x_local_structure :
            Structure of the local input tensor.
        y_local_structure :
            Structure of the local output tensor.
        P_x : Partition
            Input partition.
        P_x_to_y_overlaps : list
            List of tuples (sz, sl, partner) for each send current worker must
            perform.
        P_x_to_y_buffers : list
            List of pre-allocated send buffers for each send current worker
            must perform.
        P_y : Partition
            Input partition.
        P_y_to_x_overlaps : list
            List of tuples (sz, sl, partner) for each receive current worker
            must perform.
        P_y_to_x_buffers : list
            List of pre-allocated send buffers for each receive current worker
            must perform.
        preserve_batch : bool
            Indicates if batch size should be preserved for zero-volume outputs.

        Returns
        -------
        output :
            Output tensor.

        """

        ctx.P_union = P_union
        ctx.x_global_structure = x_global_structure
        ctx.x_local_structure = x_local_structure

        ctx.P_x = P_x
        ctx.P_x_to_y_overlaps = P_x_to_y_overlaps
        ctx.P_x_to_y_buffers = P_x_to_y_buffers

        ctx.P_y = P_y
        ctx.P_y_to_x_overlaps = P_y_to_x_overlaps
        ctx.P_y_to_x_buffers = P_y_to_x_buffers

        ctx.preserve_batch = preserve_batch

        input_requires_grad = False

        # Share the requires-grad status, so that it is preserved across the
        # transpose
        if P_union.active:
            # By design, P_x is always first in the union, so we can just take
            # rank 0's status to send
            if P_x.rank == 0:
                input_requires_grad = input.requires_grad
                P_union._comm.Bcast(np.array([1 if input_requires_grad else 0]),
                                    root=0)
            else:
                irg = np.array([0], dtype=np.int)
                P_union._comm.Bcast(irg, root=0)
                input_requires_grad = bool(irg[0] == 1)

        ctx.input_requires_grad = input_requires_grad

        requests = []

        # Default everyone to output nothing
        if preserve_batch:
            output = zero_volume_tensor(input.shape[0],
                                        dtype=x_global_structure.dtype)
        else:
            output = zero_volume_tensor(dtype=x_global_structure.dtype)

        # If I am getting data, recv my output parts
        recv_count = 0
        if P_y.active:
            for (sl, sz, partner), buff in zip(P_y_to_x_overlaps, P_y_to_x_buffers):
                if buff is not None:
                    req = P_union._comm.Irecv(buff, source=partner, tag=111)
                    requests.append(req)
                else:
                    # We add this if there is no recv so that the indices of
                    # the requests array match the indices of
                    # P_y_to_x_overlaps and P_y_to_x_buffers.
                    requests.append(MPI.REQUEST_NULL)
                recv_count += 1

        # If I have data to share, pack and send my input parts
        send_count = 0
        if P_x.active:
            input_numpy = input.detach().numpy()
            for (sl, sz, partner), buff in zip(P_x_to_y_overlaps, P_x_to_y_buffers):
                if buff is not None:
                    np.copyto(buff, input_numpy[sl].ravel())
                    req = P_union._comm.Isend(buff, dest=partner, tag=111)
                    requests.append(req)
                else:
                    # We add this for symmetry, but don't really need it.
                    requests.append(MPI.REQUEST_NULL)
                send_count += 1

        # We do this after the sends so that they can get started before local
        # allocations.
        if P_y.active:
            numpy_dtype = torch_to_numpy_dtype_dict[x_global_structure.dtype]
            output = np.zeros(y_local_structure.shape, dtype=numpy_dtype)

        # Handle the self-copy
        if P_x.active and P_y.active:
            # Find the self patch in x_to_y
            for (xsl, xsz, x2ypartner) in P_x_to_y_overlaps:
                if x2ypartner == "self":
                    for (ysl, ysz, y2xpartner) in P_y_to_x_overlaps:
                        if y2xpartner == "self":
                            np.copyto(output[ysl], input_numpy[xsl])
                            # There is only one case where this can happen
                            break
                    # There is only one case where this can happen
                    break

        # Unpack the received data as it arrives
        completed_count = 0
        while(completed_count < len(requests)):
            status = MPI.Status()
            index = MPI.Request.Waitany(requests, status)

            # In MPI, we don't get the index out if the request is an
            # instance of MPI.REQUEST_NULL, instead MPI.UNDEFINED is returned.
            if P_y.active and index < recv_count and index != MPI.UNDEFINED:
                # Unpack my output parts
                sl, sz, partner = P_y_to_x_overlaps[index]
                buff = P_y_to_x_buffers[index]
                if buff is not None:
                    sh = output[sl].shape
                    np.copyto(output[sl], buff.reshape(sh))

            completed_count += 1

        if P_y.active:
            output = torch.from_numpy(output)
            output.requires_grad = input_requires_grad

        return output

    @staticmethod
    def backward(ctx, grad_output):
        r"""Forward function of distributed transpose layer.

        This method implements the adjoint of the Jacobian of the transpose
        operation using MPI immediate-mode, non-blocking communication.

        The roles of the ``P_x`` and ``P_y`` partitions are reversed, but all
        communication across partitions occurs through the ``P_union``
        partition.

        Data is copied using ``MPI_Irecv`` and ``MPI_Isend``. As is standard
        procedure, the receives are posted first, allowing them to complete as
        they can.  Then, buffers are packed and sent.  Once all sends have
        been posted, received data is unpacked in the order that the receives
        complete.

        When the current worker is inactive in the ``P_x`` partition, it will
        output a zero-volume tensor, potentially preserving a non-zero batch

        Parameters
        ----------
        ctx :
            PyTorch context.
        grad_output : `torch.tensor`
            Input tensor.

        Returns
        -------
        output :
            Output tensor.

        """

        P_union = ctx.P_union
        x_global_structure = ctx.x_global_structure
        x_local_structure = ctx.x_local_structure

        P_x = ctx.P_x
        P_x_to_y_overlaps = ctx.P_x_to_y_overlaps
        P_x_to_y_buffers = ctx.P_x_to_y_buffers

        P_y = ctx.P_y
        P_y_to_x_overlaps = ctx.P_y_to_x_overlaps
        P_y_to_x_buffers = ctx.P_y_to_x_buffers

        preserve_batch = ctx.preserve_batch

        input_requires_grad = ctx.input_requires_grad

        requests = []

        # Default everyone to output None
        if preserve_batch:
            grad_input = zero_volume_tensor(grad_output.shape[0],
                                            dtype=x_global_structure.dtype)
        else:
            grad_input = zero_volume_tensor(dtype=x_global_structure.dtype)

        # Recv my input parts
        recv_count = 0
        if P_x.active:
            for (sl, sz, partner), buff in zip(P_x_to_y_overlaps, P_x_to_y_buffers):
                if buff is not None:
                    req = P_union._comm.Irecv(buff, source=partner, tag=113)
                    requests.append(req)
                else:
                    # We add this if there is no recv so that the indices of
                    # the requests array match the indices of
                    # P_x_to_y_overlaps and P_x_to_y_buffers.
                    requests.append(MPI.REQUEST_NULL)
                recv_count += 1

        # Pack and send my input parts
        send_count = 0
        if P_y.active:
            grad_output_numpy = grad_output.detach().numpy()
            for (sl, sz, partner), buff in zip(P_y_to_x_overlaps, P_y_to_x_buffers):
                if buff is not None:
                    np.copyto(buff, grad_output_numpy[sl].ravel())
                    req = P_union._comm.Isend(buff, dest=partner, tag=113)
                    requests.append(req)
                else:
                    # We add this for symmetry, but don't really need it.
                    requests.append(MPI.REQUEST_NULL)
                send_count += 1

        if P_x.active:
            numpy_dtype = torch_to_numpy_dtype_dict[x_global_structure.dtype]
            grad_input = np.zeros(x_local_structure.shape, dtype=numpy_dtype)

        # Handle the self-copy
        if P_y.active and P_x.active:
            # Find the self patch in x_to_y
            for (ysl, ysz, y2xpartner) in P_y_to_x_overlaps:
                if y2xpartner == "self":
                    for (xsl, xsz, x2ypartner) in P_x_to_y_overlaps:
                        if x2ypartner == "self":
                            np.copyto(grad_input[xsl], grad_output_numpy[ysl])
                            # There is only one case where this can happen
                            break
                    # There is only one case where this can happen
                    break

        # Unpack the received data as it arrives
        completed_count = 0
        while(completed_count < len(requests)):
            status = MPI.Status()
            index = MPI.Request.Waitany(requests, status)

            # In MPI, we don't get the index out if the request is an
            # instance of MPI.REQUEST_NULL, instead MPI.UNDEFINED is returned.
            if P_x.active and index < recv_count and index != MPI.UNDEFINED:
                # Unpack my output parts
                sl, sz, partner = P_x_to_y_overlaps[index]
                buff = P_x_to_y_buffers[index]
                if buff is not None:
                    sh = grad_input[sl].shape
                    # This would normally be an add into the grad_input tensor
                    # but we just created it, so a copy is sufficient.
                    np.copyto(grad_input[sl], buff.reshape(sh))

            completed_count += 1

        if P_x.active:
            grad_input = torch.from_numpy(grad_input)
            grad_input.requires_grad = input_requires_grad

        return grad_input, None, None, None, None, None, None, None, None, None, None, None
