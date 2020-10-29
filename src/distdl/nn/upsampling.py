import torch

# from distdl.nn.halo_exchange import HaloExchange
# from distdl.nn.mixins.halo_mixin import HaloMixin
# from distdl.nn.mixins.pooling_mixin import PoolingMixin
from distdl.nn.module import Module
from distdl.nn.padnd import PadNd
from distdl.utilities.slicing import assemble_slices
from distdl.utilities.torch import TensorStructure


class DistributedUpsample(Module): #, HaloMixin, UpsampleMixin):
    r"""A tensor-partitioned distributed upsampling layer.

    This class provides the user interface to a distributed upsampling
    layer, where the input (and output) tensors are partitioned in arbitrary
    dimensions.

    The base unit of work is given by the input/output tensor partition.  This
    class requires the following of the tensor partitions:

    1. :math:`P_x` over input tensor :math:`x` has shape :math:`1 \times
       p_{\text{c_in}} \times p_{d-1} \times \dots \times p_{0}`.

    The output partition, :math:`P_y`, is assumed to be the same as the
    input partition.

    The first dimension of the input/output partitions is the batch
    dimension, the second is the channel dimension, and remaining dimensions
    are feature dimensions.

    There are no learnable parameters.

    Parameters
    ----------
    P_x :
        Partition of input tensor.

    """

    def __init__(self, P_x, buffer_manager=None, *args, **kwargs):

        super(DistributedUpsample, self).__init__()

        # P_x is 1 x 1 x P_d-1 x ... x P_0
        self.P_x = P_x

        # Back-end specific buffer manager for economic buffer allocation
        if buffer_manager is None:
            buffer_manager = self._distdl_backend.BufferManager()
        elif type(buffer_manager) is not self._distdl_backend.BufferManager:
            raise ValueError("Buffer manager type does not match backend.")
        self.buffer_manager = buffer_manager

        if not self.P_x.active:
            return

        # Do this before checking serial so that the layer works properly
        # in the serial case
        # self.pool_layer = self.TorchPoolType(*args, **kwargs)

        mode = 'nearest'
        self.interp_layer = Interpolate(mode, x_starts, x_stops, y_starts, y_stops)

        # We need the halo shape, and other info, to fully populate the pad
        # and halo exchange layers.  For pad, we defer the construction to the
        # pre-forward hook.
        self.pad_layer = None

        # For the halo layer we also defer construction, so that we can have
        # the halo shape for the input.  The halo will allocate its own
        # buffers, but it needs this information at construction to be able
        # to do this in the pre-forward hook.
        self.halo_layer = None

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

    def _distdl_module_setup(self, input):
        r"""Distributed (feature) pooling module setup function.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        self._distdl_is_setup = True
        self._input_tensor_structure = TensorStructure(input[0])

        if not self.P_x.active:
            return

        # To compute the halo regions and interpolation, we need the global
        # tensor shape.  This is not available until when the input is
        # provided.
        x_global_structure = \
            self._distdl_backend.assemble_global_tensor_structure(input[0], self.P_x)




        # Using that information, we can get there rest of the halo information
        # exchange_info = self._compute_exchange_info(x_global_structure.shape,
        #                                             self.pool_layer.kernel_size,
        #                                             self.pool_layer.stride,
        #                                             self.pool_layer.padding,
        #                                             [1],  # torch pooling layers have no dilation
        #                                             self.P_x.active,
        #                                             self.P_x.shape,
        #                                             self.P_x.index)
        # halo_shape = exchange_info[0]
        # recv_buffer_shape = exchange_info[1]
        # send_buffer_shape = exchange_info[2]
        # needed_ranges = exchange_info[3]

        # Now we have enough information to instantiate the padding shim
        # self.pad_layer = PadNd(halo_shape, value=0)

        # We can also set up part of the halo layer.
        # self.halo_layer = HaloExchange(self.P_x,
        #                                halo_shape,
        #                                recv_buffer_shape,
        #                                send_buffer_shape,
        #                                buffer_manager=self.buffer_manager)

    def _distdl_module_teardown(self, input):
        r"""Distributed (channel) pooling module teardown function.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        # Reset all sub_layers
        self.pad_layer = None
        self.needed_slices = None
        self.halo_layer = None

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
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to be broadcast.

        """

        if not self.P_x.active:
            return input.clone()

        # y = self.interp(input)


        return input.clone()

        # input_padded = self.pad_layer(input)
        # input_exchanged = self.halo_layer(input_padded)
        # input_needed = input_exchanged[self.needed_slices]
        # return self.pool_layer(input_needed)
