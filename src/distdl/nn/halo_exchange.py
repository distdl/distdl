from distdl.nn.module import Module
from distdl.utilities.torch import TensorStructure


class HaloExchange(Module):

    def __init__(self, P_x, halo_shape, recv_buffer_shape, send_buffer_shape):

        super(HaloExchange, self).__init__()

        self.P_x = P_x
        self.halo_shape = halo_shape
        self.recv_buffer_shape = recv_buffer_shape
        self.send_buffer_shape = send_buffer_shape

        self.neighbor_ranks = None
        if self.P_x.active:
            self.neighbor_ranks = self.P_x.neighbor_ranks(self.P_x.rank)

        self.slices = None
        self.buffers = None

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

        # Get some types and functions from the back-end
        self.allocate_halo_exchange_buffers = self._distdl_backend.halo_exchange.allocate_halo_exchange_buffers

    def _assemble_slices(self, x_local_shape, recv_buffer_shape, send_buffer_shape):

        dim = len(x_local_shape)

        slices = []

        for i in range(dim):
            slices_i = [[], [], [], []]

            for j in range(dim):
                s = x_local_shape[j]

                lrecv_size = int(recv_buffer_shape[j, 0])
                lsend_size = int(send_buffer_shape[j, 0])
                rrecv_size = int(recv_buffer_shape[j, 1])
                rsend_size = int(send_buffer_shape[j, 1])

                # Left bulk and ghost start/stop values
                lb_start = lrecv_size
                lb_stop = lrecv_size + lsend_size
                lg_start = 0
                lg_stop = lrecv_size

                # Right bulk and ghost start/stop values
                rb_start = s - (rrecv_size + rsend_size)
                rb_stop = s - rrecv_size
                rg_start = s - rrecv_size
                rg_stop = s

                # For each dimension i, we need to define a dim-dimensional
                # rectangular prism that forms the region we are slicing for
                # the left bulk, left ghost, right bulk, and right ghost. These
                # regions are computed in a nested manner than ensures that the
                # values in the corners are correct.

                # When the dimension of the current side of the rectangular
                # prism is less than the dimension of interest for the slices,
                # we take the entire width of the tensor.
                if j < i:
                    slices_i[0].append(slice(lg_start, rg_stop, None))
                    slices_i[1].append(slice(lg_start, rg_stop, None))
                    slices_i[2].append(slice(lg_start, rg_stop, None))
                    slices_i[3].append(slice(lg_start, rg_stop, None))

                # When the dimension of the current side of the rectangular
                # prism is equal to the dimension of interest for the slices,
                # we take distinct values for the width of each bulk and ghost region.
                elif j == i:
                    slices_i[0].append(slice(lb_start, lb_stop, None))
                    slices_i[1].append(slice(lg_start, lg_stop, None))
                    slices_i[2].append(slice(rb_start, rb_stop, None))
                    slices_i[3].append(slice(rg_start, rg_stop, None))

                # Otherwise, we include only the values in the bulk.
                else:
                    slices_i[0].append(slice(lb_start, rb_stop, None))
                    slices_i[1].append(slice(lb_start, rb_stop, None))
                    slices_i[2].append(slice(lb_start, rb_stop, None))
                    slices_i[3].append(slice(lb_start, rb_stop, None))

            slices.append([tuple(x) for x in slices_i])

        return slices

    def _distdl_module_setup(self, input):

        if self.P_x.active:
            x_local_shape = input[0].shape
            self.slices = self._assemble_slices(x_local_shape, self.recv_buffer_shape, self.send_buffer_shape)
            self.buffers = self.allocate_halo_exchange_buffers(self.slices,
                                                               self.recv_buffer_shape,
                                                               self.send_buffer_shape,
                                                               input[0].dtype)

        self._distdl_is_setup = True
        self._input_tensor_structure = TensorStructure(input[0])

    def _distdl_module_teardown(self, input):

        # Reset all of the buffers and communication objects
        self.slices = None
        self.buffers = None

        # Reset any info about the input
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

    def _distdl_input_changed(self, input):
        r"""Determine if the structure of inputs has changed.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        new_tensor_structure = TensorStructure(input[0])

        return self._input_tensor_structure != new_tensor_structure

    def forward(self, input):

        Function = self._distdl_backend.autograd.halo_exchange.HaloExchangeFunction

        if not self.P_x.active:
            return input.clone()

        return Function.apply(input,
                              self.P_x,
                              self.slices,
                              self.buffers,
                              self.neighbor_ranks)
