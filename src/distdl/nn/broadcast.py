import numpy as np

from distdl.nn.module import Module


class Broadcast(Module):

    def __init__(self, P_in, P_out, transpose_src=False, transpose_dest=False):
        super(Broadcast, self).__init__()

        self.P_in = P_in
        self.P_out = P_out

        self.transpose_src = transpose_src
        self.transpose_dest = transpose_dest

        # TODO: #25  Make selection of dtype more sensible.
        self.dtype = np.float32

        self.identity = False

        # Blank partitions
        self.P_send = self._distdl_backend.Partition()
        self.P_recv = self._distdl_backend.Partition()

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_shape = None
        self._input_requires_grad = None

        # The identity case is if the partitions are of size 1,
        # or they are the same partition and neither is tranposed,
        # or they are the same partition and both are transposed.
        if self.P_in == self.P_out:
            if self.P_in.size == 1:
                self.identity = True
            elif (self.transpose_dest and self.transpose_src) or \
                 (not self.transpose_dest and not self.transpose_src):
                self.identity = True

    def _distdl_module_setup(self, input):

        # If it is not an identity, we need actual Partitions to do the work.
        if not self.identity:
            bcast_partitions = self.P_in.create_broadcast_partition_to(self.P_out,
                                                                       self.transpose_src,
                                                                       self.transpose_dest)
            self.P_send = bcast_partitions[0]
            self.P_recv = bcast_partitions[1]

        self._distdl_is_setup = True
        self._input_shape = input[0].shape
        self._input_requires_grad = input[0].requires_grad

    def _distdl_module_teardown(self, input):

        # Reset all of the buffers and communication objects
        self.P_send = self._distdl_backend.Partition()
        self.P_recv = self._distdl_backend.Partition()

        # Reset any info about the input
        self._distdl_is_setup = False
        self._input_shape = None
        self._input_requires_grad = None

    def _distdl_input_changed(self, input):

        if input[0].requires_grad != self._input_requires_grad:
            return True

        if input[0].shape != self._input_shape:
            return True

        return False

    def forward(self, input):

        Function = self._distdl_backend.autograd.broadcast.BroadcastFunction

        if self.identity:
            return input.clone()

        return Function.apply(input, self.P_send, self.P_recv, self.dtype)
