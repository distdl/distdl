import numpy as np

from distdl.backends.mpi.exchange_tensor import compute_output_tensor_structure
from distdl.nn.module import Module


class SumReduce(Module):

    def __init__(self, P_x, P_y, transpose_src=False, transpose_dest=False):
        super(SumReduce, self).__init__()

        self.P_x = P_x
        self.P_y = P_y

        self.transpose_src = transpose_src
        self.transpose_dest = transpose_dest

        # TODO: #25  Make selection of dtype more sensible.
        self.dtype = np.float32

        self.identity = False

        # Blank partitions
        self.P_send = self._distdl_backend.Partition()
        self.P_recv = self._distdl_backend.Partition()

        # Other info needed by the functions
        self.input_tensor_structure = None
        self.output_tensor_structure = None

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_shape = None
        self._input_requires_grad = None

        # The identity case is if the partitions are of size 1,
        # or they are the same partition and neither is tranposed,
        # or they are the same partition and both are transposed.
        if self.P_x == self.P_y:
            if self.P_x.size == 1:
                self.identity = True
            elif (self.transpose_dest and self.transpose_src) or \
                 (not self.transpose_dest and not self.transpose_src):
                self.identity = True

    def _distdl_module_setup(self, input):

        # If it is not an identity, we need actual Partitions to do the work.
        if not self.identity:
            reduce_partitions = self.P_x.create_reduction_partition_to(self.P_y,
                                                                       self.transpose_src,
                                                                       self.transpose_dest)
            self.P_send = reduce_partitions[0]
            self.P_recv = reduce_partitions[1]

            self.input_tensor_structure = (input[0].requires_grad,
                                           len(input[0].shape),
                                           np.array(input[0].shape, dtype=np.int))
            self.output_tensor_structure = compute_output_tensor_structure(input[0],
                                                                           self.P_send,
                                                                           self.P_recv)

        self._distdl_is_setup = True
        self._input_shape = input[0].shape
        self._input_requires_grad = input[0].requires_grad

    def _distdl_module_teardown(self, input):

        # Reset all of the buffers and communication objects
        self.P_send = self._distdl_backend.Partition()
        self.P_recv = self._distdl_backend.Partition()

        # Reset any data stored about the tensor
        self.input_tensor_structure = None
        self.output_tensor_structure = None

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

        Function = self._distdl_backend.autograd.sum_reduce.SumReduceFunction

        if self.identity:
            return input.clone()

        return Function.apply(input,
                              self.P_send,
                              self.P_recv,
                              self.input_tensor_structure,
                              self.output_tensor_structure,
                              self.dtype)
