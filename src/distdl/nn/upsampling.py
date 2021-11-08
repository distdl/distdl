import numpy as np
import torch
import torch.nn.functional as F

from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.interpolate import Interpolate
from distdl.nn.mixins.interpolate_mixin import InterpolateMixin
from distdl.nn.module import Module
from distdl.utilities.slicing import assemble_slices
from distdl.utilities.tensor_decomposition import compute_subtensor_shapes_balanced
from distdl.utilities.tensor_decomposition import compute_subtensor_start_indices
from distdl.utilities.tensor_decomposition import compute_subtensor_stop_indices
from distdl.utilities.torch import distdl_padding_to_torch_padding
from distdl.utilities.torch import TensorStructure


class DistributedUpsample(Module, InterpolateMixin):
    r"""A tensor-partitioned distributed upsampling layer.

    This class provides the user interface to a distributed upsampling
    layer, where the input (and output) tensors are partitioned in arbitrary
    dimensions.

    This class produces identical (to floating-point error) output as its
    sequential PyTorch counterpart, `torch.nn.Upsample`.

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

    Upsampling occurs over feature dimensions only.

    Parameters
    ----------
    P_x :
        Partition of input tensor.
    buffer_manager : optional
        External manager for communication buffers
    size : optional
        Desired output size.  Only one of `size` and `scale_factor` may be set.
    scale_factor : optional
        Scale-factor representing a specific scaling used to obtain
        the relationship between global input and output tensors.
    mode : string
        Interpolation mode.
    align_corners : bool
        Analogous to PyTorch UpSample's `align_corner` flag.

    """

    def __init__(self, P_x, buffer_manager=None,
                 size=None, scale_factor=None,
                 mode='linear', align_corners=False):

        super(DistributedUpsample, self).__init__()

        if mode == 'cubic':
            raise NotImplementedError('Cubic interpolation is not implemented.')

        if size is None and scale_factor is None:
            raise ValueError("One of `size` or `scale_factor` must be set.")

        if size is not None and scale_factor is not None:
            raise ValueError("Only one of `size` or `scale_factor` may be set.")

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

        self.mode = mode
        self.align_corners = align_corners

        self.size = size
        self.scale_factor = scale_factor

        # Local input and output tensor structures, defined when layer is called
        self.input_tensor_structure = TensorStructure()
        self.output_tensor_structure = TensorStructure()

        # We need the actual sizes to determine the interpolation layer
        self.interp_layer = None

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
        global_input_tensor_structure = \
            self._distdl_backend.assemble_global_tensor_structure(input[0], self.P_x)

        if self.size is None:
            global_output_tensor_shape = torch.as_tensor(global_input_tensor_structure.shape).to(torch.float64)
            global_output_tensor_shape[2:] *= self.scale_factor

            # I prefer ceil(), torch uses floor(), so we go with floor for consistency
            global_output_tensor_shape = torch.Size(torch.floor(global_output_tensor_shape).to(torch.int64))
        else:
            if len(self.size) != len(global_input_tensor_structure.shape):
                raise ValueError("Provided size does not match input tensor dimension.")
            global_output_tensor_shape = torch.Size(torch.as_tensor(self.size))
        global_output_tensor_structure = TensorStructure()
        global_output_tensor_structure.shape = global_output_tensor_shape

        # Using that information, we can get there rest of the halo information
        exchange_info = self._compute_exchange_info(self.P_x,
                                                    global_input_tensor_structure,
                                                    global_output_tensor_structure,
                                                    self.scale_factor,
                                                    self.mode,
                                                    self.align_corners)
        halo_shape = exchange_info[0]
        recv_buffer_shape = exchange_info[1]
        send_buffer_shape = exchange_info[2]
        needed_ranges = exchange_info[3]

        self.halo_shape = halo_shape

        # We can also set up part of the halo layer.
        self.halo_layer = HaloExchange(self.P_x,
                                       halo_shape,
                                       recv_buffer_shape,
                                       send_buffer_shape,
                                       buffer_manager=self.buffer_manager)

        # We have to select out the "unused" entries.  Sometimes there can
        # be "negative" halos.
        self.needed_slices = assemble_slices(needed_ranges[:, 0],
                                             needed_ranges[:, 1])

        # TODO #176: This block to compute the start and stop index of the
        # post-halo exchanged input can be cleaned up, as it is a duplicate of
        # calculation in the halo layer itself
        _slice = tuple([slice(i, i+1) for i in self.P_x.index] + [slice(None)])

        x_subtensor_shapes = compute_subtensor_shapes_balanced(global_input_tensor_structure,
                                                               self.P_x.shape)
        x_subtensor_start_indices = compute_subtensor_start_indices(x_subtensor_shapes)
        x_subtensor_stop_indices = compute_subtensor_stop_indices(x_subtensor_shapes)

        x_start_index = torch.from_numpy(x_subtensor_start_indices[_slice].squeeze())
        x_stop_index = torch.from_numpy(x_subtensor_stop_indices[_slice].squeeze())

        y_subtensor_shapes = compute_subtensor_shapes_balanced(global_output_tensor_structure,
                                                               self.P_x.shape)
        y_subtensor_start_indices = compute_subtensor_start_indices(y_subtensor_shapes)
        y_subtensor_stop_indices = compute_subtensor_stop_indices(y_subtensor_shapes)

        y_start_index = torch.from_numpy(y_subtensor_start_indices[_slice].squeeze())
        y_stop_index = torch.from_numpy(y_subtensor_stop_indices[_slice].squeeze())

        x_start_index = self._compute_needed_start(y_start_index,
                                                   global_input_tensor_structure.shape,
                                                   global_output_tensor_structure.shape,
                                                   self.scale_factor,
                                                   self.mode,
                                                   self.align_corners)

        x_stop_index = self._compute_needed_stop(y_stop_index-1,
                                                 global_input_tensor_structure.shape,
                                                 global_output_tensor_structure.shape,
                                                 self.scale_factor,
                                                 self.mode,
                                                 self.align_corners)

        self.interp_layer = Interpolate(x_start_index, x_stop_index, global_input_tensor_structure.shape,
                                        y_start_index, y_stop_index, global_output_tensor_structure.shape,
                                        scale_factor=self.scale_factor,
                                        mode=self.mode,
                                        align_corners=self.align_corners)

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

        # Compute the total padding and convert to PyTorch format
        total_padding = self.halo_shape
        torch_padding = distdl_padding_to_torch_padding(total_padding)

        if total_padding.sum() == 0:
            input_padded = input
        else:
            input_padded = F.pad(input, pad=torch_padding, mode='constant', value=0)


        input_exchanged = self.halo_layer(input_padded)
        input_needed = input_exchanged[self.needed_slices]
        y = self.interp_layer(input_needed)
        return y
