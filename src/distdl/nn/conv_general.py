import numpy as np
import torch
import torch.nn.functional as F

from distdl.nn.broadcast import Broadcast
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.mixins.conv_mixin import ConvMixin
from distdl.nn.mixins.halo_mixin import HaloMixin
from distdl.nn.module import Module
from distdl.nn.sum_reduce import SumReduce
from distdl.utilities.slicing import assemble_slices
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.slicing import range_index
from distdl.utilities.torch import TensorStructure
from distdl.utilities.torch import distdl_padding_to_torch_padding
from distdl.utilities.torch import zero_volume_tensor


class DistributedGeneralConvBase(Module, HaloMixin, ConvMixin):
    r"""A generally partitioned distributed convolutional layer.

    This class provides the user interface to a distributed convolutional
    layer, where the input (and output) tensors are partitioned in both the
    channel and feature dimensions.

    The base unit of work is given by the weight tensor partition.  This class
    requires the following of the tensor partitions:

    1. :math:`P_x` over input tensor :math:`x` has shape :math:`1 \times
       P_{\text{c_in}} \times P_{d-1} \times \dots \times P_0`.
    2. :math:`P_y` over input tensor :math:`y` has shape :math:`1 \times
       P_{\text{c_out}} \times P_{d-1} \times \dots \times P_0`.
    3. :math:`P_W` over weight tensor :math:`W` has shape
       :math:`P_{\text{c_out}} \times P_{\text{c_in}}  \times P_{d-1} \times
       \dots \times P_0`.

    The first dimension of the input and output partitions is the batch
    dimension,the second is the channel dimension, and remaining dimensions
    are feature dimensions.

    The bias term does not have its own partition.  It is stored in the first
    "column" of :math:`P_w`, that is a :math:`P_{\text{c_out}} \times 1`
    subpartition.

    Parameters
    ----------
    P_x :
        Partition of input tensor.
    P_y :
        Partition of output tensor.
    P_w :
        Partition of the weight tensor.
    in_channels :
        Number of channels in the *global* input tensor.
    out_channels :
        Number of channels in the *global* output tensor.
    kernel_size :
        (int or tuple)
        Size of the convolving kernel
    stride :
        (int or tuple, optional)
        Stride of the convolution. Default: 1
    padding :
        (int or tuple, optional)
        Zero-padding added to both sides of the input. Default: 0
    padding_mode :
        (string, optional)
        'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    dilation :
        (int or tuple, optional)
        Spacing between kernel elements. Default: 1
    groups :
        (int, optional)
        Number of blocked connections from input channels to output channels. Default: 1
    bias :
        (bool, optional)
        If True, adds a learnable bias to the output. Default: True
    buffer_manager :
        (BufferManager, optional)
        DistDL BufferManager. Default: None

    """

    # Convolution class for base unit of work.
    TorchConvType = None
    # Number of dimensions of a feature
    num_dimensions = None

    def __init__(self, P_x, P_y, P_w,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 padding_mode='zeros',
                 dilation=1,
                 groups=1,
                 bias=True,
                 buffer_manager=None):

        super(DistributedGeneralConvBase, self).__init__()

        # P_x is 1    x P_ci x P_d-1 x ... x P_0
        self.P_x = P_x
        # P_y is 1    x P_co x P_d-1 x ... x P_0
        self.P_y = P_y
        # P_w is P_co x P_ci x P_d-1 x ... x P_0
        self.P_w = P_w

        # Back-end specific buffer manager for economic buffer allocation
        if buffer_manager is None:
            buffer_manager = self._distdl_backend.BufferManager()
        elif type(buffer_manager) is not self._distdl_backend.BufferManager:
            raise ValueError("Buffer manager type does not match backend.")
        self.buffer_manager = buffer_manager

        # Even inactive workers need some partition union
        self.P_union = self._distdl_backend.Partition()
        if not (self.P_x.active or
                self.P_y.active or
                self.P_w.active):
            return

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._expand_parameter(kernel_size)
        self.stride = self._expand_parameter(stride)
        self.padding = self._expand_parameter(padding)
        self.padding_mode = padding_mode
        self.dilation = self._expand_parameter(dilation)
        self.groups = groups
        self.use_bias = bias

        # This guarantees that P_union rank 0 has the kernel size, stride,
        # padding, and dilation factors
        P_union_temp = P_w.create_partition_union(P_x)
        self.P_union = P_union_temp.create_partition_union(P_y)

        # Release the temporary resources
        P_union_temp.deactivate()

        # Ensure that all workers have the full size and structure of P_w
        P_w_shape = None
        if self.P_union.rank == 0:
            P_w_shape = np.array(P_w.shape, dtype=np.int)
        P_w_shape = self.P_union.broadcast_data(P_w_shape, root=0)

        P_co = P_w_shape[0]
        P_ci = P_w_shape[1]
        P_channels = [P_co, P_ci]

        # Ensure that P_x and P_w are correctly aligned.  We also produce a
        # new P_x that is shaped like 1 x P_ci x P_d-1 x ... x P_0, to assist
        # with broadcasts.
        P_x_new_shape = []
        if self.P_x.active:
            if(np.any(P_x.shape[2:] != P_w_shape[2:])):
                raise ValueError("Spatial components of P_x and P_w must match.")
            if P_w_shape[1] != P_x.shape[1]:
                raise ValueError("Index 2 of P_w dimension must match input channel partition.")
            P_x_new_shape = list(P_x.shape)
            P_x_new_shape.insert(1, 1)
            # Currently a hack, removing the batch dimension because P_w does
            # not have one. This is OK because we assume there are no partitions
            # in the batch dimension.
            P_x_new_shape = np.asarray(P_x_new_shape[1:], dtype=int)

        # For the purposes of this layer, we re-cast P_x to have the extra
        # dimension.  This has no impact outside of the layer or on the results.
        self.P_x = self.P_x.create_cartesian_topology_partition(P_x_new_shape)

        # Ensure that P_y and P_w are correctly aligned.  We also produce a
        # new P_y that is shaped like 1 x P_ci x P_d-1 x ... x P_0, to assist
        # with broadcasts.
        P_y_new_shape = []
        if self.P_y.active:
            if(np.any(P_y.shape[2:] != P_w_shape[2:])):
                raise ValueError("Spatial components of P_y and P_w must match.")
            if P_w_shape[0] != P_y.shape[1]:
                raise ValueError("Index 1 of P_w dimension must match output channel partition.")
            P_y_new_shape = list(P_y.shape)
            P_y_new_shape.insert(2, 1)
            # Currently a hack, removing the batch dimension because P_w does
            # not have one. This is OK because we assume there are no partitions
            # in the batch dimension.
            P_y_new_shape = np.asarray(P_y_new_shape[1:], dtype=int)

        # For the purposes of this layer, we re-cast P_x to have the extra
        # dimension.  This has no impact outside of the layer or on the results.
        self.P_y = self.P_y.create_cartesian_topology_partition(P_y_new_shape)

        P_spatial = P_w_shape[2:]

        self.serial = self.P_w.size == 1

        if self.serial:
            self.conv_layer = self.TorchConvType(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 padding_mode=self.padding_mode,
                                                 dilation=self.dilation,
                                                 groups=self.groups,
                                                 bias=self.use_bias)
            self.weight = self.conv_layer.weight
            self.bias = self.conv_layer.bias
            return

        # Need to figure out any padding necessary to handle global padding.
        # This is only on the input tensor.  The convolution will not use
        # any implicit padding, so the work partition does not need it.
        if self.P_x.active:
            dims = len(self.P_x.shape)

            # We will be using global padding to compute local padding,
            # so expand it to a numpy array
            global_padding = np.pad(self.padding,
                                    pad_width=(dims-len(self.padding), 0),
                                    mode='constant',
                                    constant_values=0)
            self.global_padding = global_padding

            pad_left_right = self.global_padding.reshape((dims, 1)) + np.zeros((dims, 2), dtype=np.int)
            self.local_padding = self._compute_local_padding(pad_left_right)

        # Workers can either store the learnable weights and bias, or they
        # need copies of it.
        self.receives_weight = False
        self.stores_weight = False
        self.receives_bias = False
        self.stores_bias = False

        # Determine root partitions, initialize weights there
        if self.P_w.active:
            # All of P_w always receives the weight
            self.receives_weight = True

            # This subset is taken to be the origin of the spartial component
            w_root_subset = []
            for i, c in enumerate(range_index(P_w.shape)):
                c = np.asarray(c)
                # Find the P_co x P_ci x 1 x ... x 1 subset to store the weights
                if np.all(c[2:] == 0):
                    w_root_subset.append(i)

            P_wr_base = self.P_w.create_partition_inclusive(w_root_subset)
            # ones are needed so the broadcast will work
            self.P_wr = P_wr_base.create_cartesian_topology_partition([P_co, P_ci] + [1]*len(P_spatial))
            self.stores_weight = self.P_wr.active

            # Release temporary resources
            P_wr_base.deactivate()

            b_subset = []
            for i, c in enumerate(range_index(P_w.shape)):
                c = np.asarray(c)
                # Find the P_co x 1 x P_0 x ... x P_D-1 subset that needs
                # biases in its calculation. This is everywhere that the input
                # channels is rank 0.
                if c[1] == 0:
                    b_subset.append(i)

            P_b_base = self.P_w.create_partition_inclusive(b_subset)
            self.P_b = P_b_base.create_cartesian_topology_partition([P_co] + [1] + list(P_spatial))
            self.receives_bias = self.P_b.active and bias

            # Release temporary resources
            P_b_base.deactivate()

            # Now find the subset of _that_ which actually stores the
            # learnable parameter.
            b_root_subset = []
            for i, c in enumerate(range_index(P_w.shape)):
                c = np.asarray(c)
            # Find the P_co x 1 x 1 x ... x 1 subset to store the biases
                if np.all(c[1:] == 0):
                    b_root_subset.append(i)

            P_br_base = self.P_w.create_partition_inclusive(b_root_subset)
            # ones are needed so the broadcast will work
            self.P_br = P_br_base.create_cartesian_topology_partition([P_co] + [1] + [1]*len(P_spatial))
            self.stores_bias = self.P_br.active and bias

            # Release temporary resources
            P_br_base.deactivate()

            # Correct the input arguments based on local properties
            # This ensures that the in and out channels are correctly shared.
            local_co, local_ci = compute_subshape(P_channels,
                                                  P_w.index[0:2],
                                                  [out_channels, in_channels])
            self.conv_layer = self.TorchConvType(in_channels=local_ci,
                                                 out_channels=local_co,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=0,
                                                 padding_mode='zeros',
                                                 dilation=self.dilation,
                                                 groups=groups,
                                                 bias=self.receives_bias)

            # If we store the weight it is a learnable parameter iff it is
            # learnable by default in the layer, which it is.
            if self.stores_weight:
                self.weight = torch.nn.Parameter(self.conv_layer.weight.detach())
            else:
                self.register_buffer('weight', zero_volume_tensor())
            # This always exists so we can copy the property
            self.weight.requires_grad = self.conv_layer.weight.requires_grad

            # https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677/2
            new_weight = self.conv_layer.weight.detach() * 0
            new_weight.requires_grad = self.conv_layer.weight.requires_grad
            del self.conv_layer.weight
            self.conv_layer.weight = new_weight

            # If we store the bias, it is a learnable parameter iff it is
            # learnable by default in the layer, which is only true if it
            # exists.
            if self.stores_bias:
                self.bias = torch.nn.Parameter(self.conv_layer.bias.detach())
            else:
                self.register_buffer('bias', zero_volume_tensor())
            # This does not always exist, but when it does we can copy the
            # property.
            if self.receives_bias:
                self.bias.requires_grad = self.conv_layer.bias.requires_grad

                # https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677/2
                new_bias = self.conv_layer.bias.detach() * 0
                new_bias.requires_grad = self.conv_layer.bias.requires_grad
                del self.conv_layer.bias
                self.conv_layer.bias = new_bias

        # Now we need to share the kernel structure.  The size of the kernel
        # is always the spatial dimensions.
        self.conv_kernel_size = None
        self.conv_stride = None
        self.conv_padding = None
        self.conv_dilation = None

        # By construction, rank 0 of the union should always have all of this
        # information, because it will always construct a local conv layer. We
        # rely on the local conv layer to properly fill out this information
        # from the defaults.  This info is required for all workers on the
        # input and output partitions because it is needed to construct the
        # halos.  Rank 0 in the union shares it with everyone.
        if self.P_union.rank == 0:
            self.conv_kernel_size = np.array(self.conv_layer.kernel_size, dtype=np.int)
            self.conv_stride = np.array(self.conv_layer.stride, dtype=np.int)
            self.conv_padding = np.array(self.conv_layer.padding, dtype=np.int)
            self.conv_dilation = np.array(self.conv_layer.dilation, dtype=np.int)
        self.conv_kernel_size = self.P_union.broadcast_data(self.conv_kernel_size, root=0)
        self.conv_stride = self.P_union.broadcast_data(self.conv_stride, root=0)
        self.conv_padding = self.P_union.broadcast_data(self.conv_padding, root=0)
        self.conv_dilation = self.P_union.broadcast_data(self.conv_dilation, root=0)

        # We need to be able to remove some data from the input to the conv
        # layer but again need to defer.
        self.needed_slices = None

        # For the halo layer we also defer construction, so that we can have
        # the halo shape for the input.  The halo will allocate its own
        # buffers, but it needs this information at construction to be able
        # to do this in the pre-forward hook.
        self.halo_layer = None

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

        # Some layers, those that require no information about the input
        # tensor to setup, can be built now.
        if P_w.active:
            self.w_broadcast = Broadcast(self.P_wr, self.P_w, preserve_batch=False)

        if self.receives_bias or self.stores_bias:
            self.b_broadcast = Broadcast(self.P_br, self.P_b, preserve_batch=False)

        self.x_broadcast = Broadcast(self.P_x, self.P_w, preserve_batch=True)
        self.y_sum_reduce = SumReduce(self.P_w, self.P_y, preserve_batch=True)

    def _expand_parameter(self, param):
        # If the given input is not of size num_dimensions, expand it so.
        # If not possible, raise an exception.
        param = np.atleast_1d(param)
        if len(param) == 1:
            param = np.ones(self.num_dimensions, dtype=int) * param[0]
        elif len(param) == self.num_dimensions:
            pass
        else:
            raise ValueError('Invalid parameter: ' + str(param))
        return tuple(param)

    def _distdl_module_setup(self, input):
        r"""Distributed (general) convolution module setup function.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        if not (self.P_x.active or
                self.P_y.active or
                self.P_w.active):
            return

        if self.serial:
            return

        # To compute the halo regions, we need the global tensor shape.  This
        # is not available until when the input is provided.
        x_global_structure = \
            self._distdl_backend.assemble_global_tensor_structure(input[0],
                                                                  self.P_x,
                                                                  self.P_union)

        if self.P_x.active:
            x_local_structure = TensorStructure(input[0])
            x_global_shape = x_global_structure.shape
            x_local_shape = x_local_structure.shape
            x_global_shape_after_pad = x_global_shape + 2*self.global_padding
            x_local_shape_after_pad = x_local_shape + np.sum(self.local_padding, axis=1, keepdims=False)
            x_local_structure_after_pad = TensorStructure(input[0])
            x_local_structure_after_pad.shape = x_local_shape_after_pad

            # We need to compute the halos with respect to the explicit padding.
            # So, we assume the padding is already added, then compute the halo regions.
            compute_subtensor_shapes_unbalanced = \
                self._distdl_backend.tensor_decomposition.compute_subtensor_shapes_unbalanced
            subtensor_shapes = \
                compute_subtensor_shapes_unbalanced(x_local_structure_after_pad, self.P_x)

            # Using that information, we can get there rest of the halo information
            exchange_info = self._compute_exchange_info(x_global_shape_after_pad,
                                                        self.kernel_size,
                                                        self.stride,
                                                        self._expand_parameter(0),
                                                        self.dilation,
                                                        self.P_x.active,
                                                        self.P_x.shape,
                                                        self.P_x.index,
                                                        subtensor_shapes=subtensor_shapes)
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

        self._distdl_is_setup = True
        self._input_tensor_structure = TensorStructure(input[0])

    def _distdl_module_teardown(self, input):
        r"""Distributed (channel) convolution module teardown function.

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

    def _compute_local_padding(self, padding):
        r"""
        Computes the amount of explicit padding required on the current rank,
        given the global padding.

        """
        if self.P_x.active:
            should_pad_left = [k == 0 for k in self.P_x.index]
            should_pad_right = [k == d-1 for k, d in zip(self.P_x.index, self.P_x.shape)]
            should_pad = np.stack((should_pad_left, should_pad_right), axis=1)
            local_padding = np.where(should_pad, padding, 0)
            return local_padding
        else:
            return None

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to be broadcast.

        """

        if not (self.P_x.active or
                self.P_y.active or
                self.P_w.active):
            return input.clone()

        if self.serial:
            return self.conv_layer(input)

        x = input

        if self.P_x.active:
            # Compute the total padding and convert to PyTorch format
            total_padding = self.local_padding + self.halo_shape
            torch_padding = distdl_padding_to_torch_padding(total_padding)

            if total_padding.sum() == 0:
                input_padded = input
            else:
                pad_mode = 'constant' if self.padding_mode == 'zeros' else self.padding_mode
                input_padded = F.pad(input, pad=torch_padding, mode=pad_mode, value=0)

            input_exchanged = self.halo_layer(input_padded)
            input_needed = input_exchanged[self.needed_slices]
            x = input_needed

        # Weights always received
        if self.P_w.active:
            w = self.w_broadcast(self.weight)
            self.conv_layer.weight = w

        # Biases only received in some places
        if self.receives_bias or self.stores_bias:
            b = self.b_broadcast(self.bias)
            self.conv_layer.bias = b

        x = self.x_broadcast(x)

        # assert 0
        if self.P_w.active:
            x = self.conv_layer(x)

        y = self.y_sum_reduce(x)

        return y


class DistributedGeneralConv1d(DistributedGeneralConvBase):
    r"""A general distributed 1d convolutional layer.

    """

    TorchConvType = torch.nn.Conv1d
    num_dimensions = 1


class DistributedGeneralConv2d(DistributedGeneralConvBase):
    r"""A general distributed 2d convolutional layer.

    """

    TorchConvType = torch.nn.Conv2d
    num_dimensions = 2


class DistributedGeneralConv3d(DistributedGeneralConvBase):
    r"""A general distributed 3d convolutional layer.

    """

    TorchConvType = torch.nn.Conv3d
    num_dimensions = 3
