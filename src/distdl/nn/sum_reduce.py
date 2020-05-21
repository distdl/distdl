import numpy as np
import torch
from mpi4py import MPI


class SumReduceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_common, P_in, P_out, out_root, dtype):

        # There is no reason to allow users to select the in root.
        in_root = 0

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

        if P_in.active:

            input_numpy = input.detach().numpy()

            tensor_dim = np.array(len(input_numpy.shape), dtype=np.int)
            tensor_sizes = np.array(input_numpy.shape, dtype=np.int)

            # All input ranks need intermediate storage for the reduction.
            reduced_data = np.zeros(shape=input_numpy.shape,
                                    dtype=input_numpy.dtype)

            # So that we don't have to share this information during the
            # adjoint phase, save it for later.
            ctx.tensor_sizes = tensor_sizes
            ctx.requires_grad = input.requires_grad

            # The input root has to collect the result of the sum reduction.
            # It also has to share the tensor size information with the output
            # root. We share the information first so that the output can
            # prepare to receive the result of the reduction while the
            # reduction is happening.
            if P_in.rank == in_root:

                partner = np.where(P_common_to_P_out == out_root)[0][0]

                # Share the requires_grad status with the output partition
                req = P_common.comm.isend(input.requires_grad,
                                          dest=partner, tag=2231)
                requests.append(req)

                # The output tensors do not know the size or dimension, so we
                # must share that information first.
                req = P_common.comm.Isend(tensor_dim, dest=partner, tag=2232)
                requests.append(req)

                req = P_common.comm.Isend(tensor_sizes, dest=partner, tag=2233)
                requests.append(req)

                # Now we can perform the sum reduction
                P_in.comm.Reduce(input_numpy, reduced_data,
                                 root=in_root, op=MPI.SUM)

                # And we can share the actual data
                req = P_common.comm.Isend(input_numpy, dest=partner, tag=2234)
                requests.append(req)

            # The other input ranks only do the reduction
            else:
                P_in.comm.Reduce(input_numpy, reduced_data,
                                 root=in_root, op=MPI.SUM)

        # The root rank on the output partition needs to get the basic
        # tensor details from the reduction root.
        if P_out.active and P_out.rank == out_root:
            partner = np.where(P_common_to_P_in == in_root)[0][0]

            req = P_common.comm.irecv(source=partner, tag=2231)
            requires_grad = req.wait()

            tensor_dim = np.zeros(1, dtype=np.int)
            req = P_common.comm.Irecv(tensor_dim, source=partner, tag=2232)
            req.Wait()

            tensor_sizes = np.zeros(tensor_dim, dtype=np.int)
            req = P_common.comm.Irecv(tensor_sizes, source=partner, tag=2233)
            req.Wait()

            # Allocate and receive the output tensor
            output = np.zeros(tensor_sizes, dtype=dtype)
            req = P_common.comm.Irecv(output, source=partner, tag=2234)
            req.Wait()

            # Only P_out root has data to return
            output = torch.tensor(output, requires_grad=requires_grad)
        else:
            # Everyone else has nothing
            output = None

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

        # The output partition performs the adjoint of the sum reduction, a
        # broadcast.  The root rank of the output partition sends data to
        # the root rank of the input partition, where it can be broadcast to
        # the rest of the input partition.
        if P_out.active and P_out.rank == out_root:

            grad_output_numpy = grad_output.detach().numpy()

            partner = np.where(P_common_to_P_in == in_root)[0][0]
            req = P_common.comm.Isend(grad_output_numpy, dest=partner, tag=2235)
            requests.append(req)

        # This allows all ranks to use the same exit path, so that we can be
        # sure that all requests have cleared.
        grad_input = None

        if P_in.active:

            tensor_sizes = ctx.tensor_sizes
            requires_grad = ctx.requires_grad

            grad_input = np.zeros(tensor_sizes, dtype=dtype)

            # Only the root rank of the input partition receives the data.
            # Then, it broadcasts it to the rest of the partition.
            if P_in.rank == in_root:

                partner = np.where(P_common_to_P_out == out_root)[0][0]
                req = P_common.comm.Irecv(grad_input, source=partner, tag=2235)
                req.Wait()

            # Everyone can then complete the broadcast
            P_in.comm.Bcast(grad_input, root=out_root)

            grad_input = torch.tensor(grad_input, requires_grad=requires_grad)

        # Ensure all sends have finished.
        MPI.Request.Waitall(requests)

        return grad_input, None, None, None, None, None


class SumReduce(torch.nn.Module):

    def __init__(self, P_in, P_out, out_root=0):
        super(SumReduce, self).__init__()

        self.P_in = P_in
        self.P_out = P_out

        # Find the common partition between the input and output partitions
        P_common = P_in.common_ancestor(P_out)
        if P_common is None:
            raise Exception()
        self.P_common = P_common

        self.out_root = out_root

        # TODO: #25  Make selection of dtype more sensible.
        self.dtype = np.float32

    def forward(self, input):
        return SumReduceFunction.apply(input,
                                       self.P_common, self.P_in, self.P_out,
                                       self.out_root, self.dtype)
