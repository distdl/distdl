import numpy as np
import torch
from mpi4py import MPI


class BroadcastFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_common, P_in, P_out, in_root, dtype):

        # There is no reason to allow users to select the out root.
        out_root = 0

        ctx.P_common = P_common
        ctx.P_in = P_in
        ctx.P_out = P_out
        ctx.in_root = in_root
        ctx.out_root = out_root
        ctx.dtype = dtype

        # We need to move data between two potentially disjoint partitions,
        # so we have to find a common mapping, which requires the common
        # partition.
        P_common_to_P_in = P_in.map_from_ancestor(P_common)
        P_common_to_P_out = P_out.map_from_ancestor(P_common)

        # Collect at least the send requests so they can resolve out of order.
        requests = []

        # Only the root rank in the active part of P_in has the data
        if P_in.active and P_in.rank == in_root:

            input_numpy = input.detach().numpy()

            partner = np.where(P_common_to_P_out == out_root)[0][0]

            # Share the requires_grad status with the output partition
            req = P_common.comm.isend(input.requires_grad,
                                      dest=partner, tag=1231)
            requests.append(req)

            # The output tensors do not know the size or dimension, so we must
            # share that information first.
            tensor_dim = np.array(len(input_numpy.shape), dtype=np.int)
            req = P_common.comm.Isend(tensor_dim, dest=partner, tag=1232)
            requests.append(req)

            tensor_sizes = np.array(input_numpy.shape, dtype=np.int)
            req = P_common.comm.Isend(tensor_sizes, dest=partner, tag=1233)
            requests.append(req)

            # Once they have the sizes, we can share the actual data
            req = P_common.comm.Isend(input_numpy, dest=partner, tag=1234)
            requests.append(req)

            # So that we don't have to share this information during the
            # adjoint phase, save it for later.
            ctx.tensor_sizes = tensor_sizes
            ctx.requires_grad = input.requires_grad

        # This allows all ranks to use the same exit path, so that we can be
        # sure that all requests have cleared.
        output = None

        if P_out.active:

            requires_grad = None

            # The root rank on the output partition needs to get the basic
            # tensor details from the broadcast root, and share them with the
            # remainder of the output partition.  Each of the requests must
            # be resolved in order, unfortunately.
            if P_out.rank == out_root:
                partner = np.where(P_common_to_P_in == in_root)[0][0]

                req = P_common.comm.irecv(source=partner, tag=1231)
                requires_grad = req.wait()
                requires_grad = P_out.comm.bcast(requires_grad, root=out_root)

                tensor_dim = np.zeros(1, dtype=np.int)
                req = P_common.comm.Irecv(tensor_dim, source=partner, tag=1232)
                req.Wait()
                P_out.comm.Bcast(tensor_dim, root=out_root)

                tensor_sizes = np.zeros(tensor_dim, dtype=np.int)
                req = P_common.comm.Irecv(tensor_sizes, source=partner, tag=1233)
                req.Wait()
                P_out.comm.Bcast(tensor_sizes, root=out_root)

                # Allocate and receive the output tensor
                output = np.zeros(tensor_sizes, dtype=dtype)
                req = P_common.comm.Irecv(output, source=partner, tag=1234)
                req.Wait()

            else:
                requires_grad = P_out.comm.bcast(requires_grad, root=out_root)

                tensor_dim = np.zeros(1, dtype=np.int)
                P_out.comm.Bcast(tensor_dim, root=out_root)

                tensor_sizes = np.zeros(tensor_dim, dtype=np.int)
                P_out.comm.Bcast(tensor_sizes, root=out_root)

                # Only allocate the output tensor.  Data will come in the
                # broadcast, next.
                output = np.zeros(tensor_sizes, dtype=dtype)

            P_out.comm.Bcast(output, root=out_root)

            # Active P_outs have real data to return
            output = torch.tensor(output, requires_grad=requires_grad)

        # Ensure that all send requests have cleared.
        MPI.Request.Waitall(requests)

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
            requires_grad = ctx.requires_grad

            grad_input = np.zeros(tensor_sizes, dtype=dtype)

            partner = np.where(P_common_to_P_out == out_root)[0][0]
            req = P_common.comm.Irecv(grad_input, source=partner, tag=1235)
            req.Wait()

            grad_input = torch.tensor(grad_input, requires_grad=requires_grad)

        # Ensure all sends have finished.
        MPI.Request.Waitall(requests)

        return grad_input, None, None, None, None, None


class Broadcast(torch.nn.Module):

    def __init__(self, P_in, P_out, in_root=0):
        super(Broadcast, self).__init__()

        self.P_in = P_in
        self.P_out = P_out

        # Find the common partition between the input and output partitions
        P_common = P_in.common_ancestor(P_out)
        if P_common is None:
            raise Exception()
        self.P_common = P_common

        self.in_root = in_root

        # TODO: #25  Make selection of dtype more sensible.
        self.dtype = np.float32

    def forward(self, input):
        return BroadcastFunction.apply(input,
                                       self.P_common, self.P_in, self.P_out,
                                       self.in_root, self.dtype)
