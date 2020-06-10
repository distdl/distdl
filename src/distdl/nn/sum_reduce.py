import numpy as np

from distdl.nn.module import Module


class SumReduce(Module):

    def __init__(self, P_in, P_out, transpose_src=False, transpose_dest=False):
        super(SumReduce, self).__init__()

        self.P_in = P_in
        self.P_out = P_out

        self.transpose_src = transpose_src
        self.transpose_dest = transpose_dest

        # TODO: #25  Make selection of dtype more sensible.
        self.dtype = np.float32

        self.identity = False

        # The identity case is if the partitions are of size 1,
        # or they are the same partition and neither is tranposed,
        # or they are the same partition and both are transposed.
        if P_in == P_out:
            if P_in.size == 1:
                self.identity = True
            elif (transpose_dest and transpose_src) or \
                 (not transpose_dest and not transpose_src):
                self.identity = True

        # We do the actual work if it is not an identity
        if not self.identity:
            reduce_partitions = P_in.create_reduction_partition_to(P_out,
                                                                   transpose_src,
                                                                   transpose_dest)
            self.P_send = reduce_partitions[0]
            self.P_recv = reduce_partitions[1]

    def forward(self, input):

        Function = self._distdl_backend.autograd.sum_reduce.SumReduceFunction

        if self.identity:
            return input.clone()

        return Function.apply(input, self.P_send, self.P_recv, self.dtype)
