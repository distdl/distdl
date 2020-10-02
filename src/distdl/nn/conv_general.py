import numpy as np
import torch

from distdl.nn.broadcast import Broadcast
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.mixins.conv_mixin import ConvMixin
from distdl.nn.mixins.halo_mixin import HaloMixin
from distdl.nn.module import Module
from distdl.nn.padnd import PadNd
from distdl.nn.sum_reduce import SumReduce
from distdl.nn.unpadnd import UnpadNd
from distdl.utilities.slicing import assemble_slices
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.slicing import range_index
from distdl.utilities.torch import TensorStructure
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
    bias : bool
        Indicates if a bias term should be used.

    """

    # Convolution class for base unit of work.
    TorchConvType = None

    def __init__(self, P_x, P_y, P_w,
                 in_channels=1, out_channels=1,
                 bias=True,
                 *args, **kwargs):

        super(DistributedGeneralConvBase, self).__init__()

        # P_x is 1    x P_ci x P_d-1 x ... x P_0
        self.P_x = P_x
        # P_y is 1    x P_co x P_d-1 x ... x P_0
        self.P_y = P_y
        # P_w is P_co x P_ci x P_d-1 x ... x P_0
        self.P_w = P_w

        # Even inactive workers need some partition union
        self.P_union = self._distdl_backend.Partition()
        if not (self.P_x.active or
                self.P_y.active or
                self.P_w.active):
            return

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

        self.serial = False
        if self.P_w.size == 1:
            self.serial = True
            self.conv_layer = self.TorchConvType(*args, **kwargs)
            return

        # Workers can either store the learnable weight, or they need copies
        # of it.
        self.receives_weight = False
        self.stores_weight = False

        # Workers can either store the learnable bias, or they need copies of
        # it.
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
            local_kwargs = {}
            local_kwargs.update(kwargs)

            # Do this before checking serial so that the layer works properly
            # in the serial case
            local_out_channels, local_in_channels = compute_subshape(P_channels,
                                                                     P_w.index[0:2],
                                                                     [out_channels, in_channels])
            local_kwargs["in_channels"] = local_in_channels
            local_kwargs["out_channels"] = local_out_channels
            local_kwargs["bias"] = self.receives_bias

            self.conv_layer = self.TorchConvType(*args, **local_kwargs)

            # If we store the weight it is a learnable parameter iff it is
            # learnable by default in the layer, which it is.
            if self.stores_weight:
                self._weight = torch.nn.Parameter(self.conv_layer.weight.detach())
            else:
                self._weight = zero_volume_tensor()
            # This always exists so we can copy the property
            self._weight.requires_grad = self.conv_layer.weight.requires_grad

            # https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677/2
            new_weight = self.conv_layer.weight.detach() * 0
            new_weight.requires_grad = self.conv_layer.weight.requires_grad
            del self.conv_layer.weight
            self.conv_layer.weight = new_weight

            # If we store the bias, it is a learnable parameter iff it is
            # learnable by default in the layer, which is only true if it
            # exists.
            if self.stores_bias:
                self._bias = torch.nn.Parameter(self.conv_layer.bias.detach())
            else:
                self._bias = zero_volume_tensor()
            # This does not always exist, but when it does we can copy the
            # property.
            if self.receives_bias:
                self._bias.requires_grad = self.conv_layer.bias.requires_grad

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

        # We need the halo shape, and other info, to fully populate the pad,
        # halo exchange, and unpad layers.  For pad and unpad, we defer their
        # construction to the pre-forward hook.
        self.pad_layer = None
        self.unpad_layer = None

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
            # Using that information, we can get there rest of the halo
            # information
            exchange_info = self._compute_exchange_info(x_global_structure.shape,
                                                        self.conv_kernel_size,
                                                        self.conv_stride,
                                                        self.conv_padding,
                                                        self.conv_dilation,
                                                        self.P_x.active,
                                                        self.P_x.shape,
                                                        self.P_x.index)
            halo_shape = exchange_info[0]
            recv_buffer_shape = exchange_info[1]
            send_buffer_shape = exchange_info[2]
            needed_ranges = exchange_info[3]

            # Now we have enough information to instantiate the padding shim
            self.pad_layer = PadNd(halo_shape, value=0)

            # We can also set up part of the halo layer.
            self.halo_layer = HaloExchange(self.P_x,
                                           halo_shape,
                                           recv_buffer_shape,
                                           send_buffer_shape)

            # We have to select out the "unused" entries.
            self.needed_slices = assemble_slices(needed_ranges[:, 0],
                                                 needed_ranges[:, 1])

        # The output has to do some unpadding
        if self.P_y.active:

            # This is safe because there are never halos on the channel or batch
            # dimensions.  Therefore, because we assume that the spatial partition
            # of P_x and P_y is the same, then the halo shape this will
            # compute will also be the same, even though the output feature
            # shape may be different.
            exchange_info = self._compute_exchange_info(x_global_structure.shape,
                                                        self.conv_kernel_size,
                                                        self.conv_stride,
                                                        self.conv_padding,
                                                        self.conv_dilation,
                                                        self.P_y.active,
                                                        self.P_y.shape,
                                                        self.P_y.index)
            y_halo_shape = exchange_info[0]

            # Unpad shape is padding in the dimensions where we have a halo,
            # otherwise 0
            conv_padding = np.concatenate(([0, 0], self.conv_padding))
            unpad_shape = []
            for pad, halo in zip(conv_padding, y_halo_shape):
                unpad_shape.append(np.where(halo > 0, pad, 0))
            unpad_shape = np.asarray(unpad_shape)

            self.unpad_layer = UnpadNd(unpad_shape, value=0)

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
        self.pad_layer = None
        self.unpad_layer = None
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

        if not (self.P_x.active or
                self.P_y.active or
                self.P_w.active):
            return input.clone()

        if self.serial:
            return self.conv_layer(input)

        x = input
        if self.P_x.active:
            x = self.pad_layer(x)
            x = self.halo_layer(x)
            x = x[self.needed_slices]

        # Weights always received
        if self.P_w.active:
            w = self.w_broadcast(self._weight)
            self.conv_layer.weight = w

        # Biases only received in some places
        if self.receives_bias or self.stores_bias:
            b = self.b_broadcast(self._bias)
            self.conv_layer.bias = b

        x = self.x_broadcast(x)

        if self.P_w.active:
            x = self.conv_layer(x)

        y = self.y_sum_reduce(x)

        if self.P_y.active:
            y = self.unpad_layer(y)

        return y


class DistributedGeneralConv1d(DistributedGeneralConvBase):
    r"""A general distributed 1d convolutional layer.

    """

    TorchConvType = torch.nn.Conv1d


class DistributedGeneralConv2d(DistributedGeneralConvBase):
    r"""A general distributed 2d convolutional layer.

    """

    TorchConvType = torch.nn.Conv2d


class DistributedGeneralConv3d(DistributedGeneralConvBase):
    r"""A general distributed 3d convolutional layer.

    """

    TorchConvType = torch.nn.Conv3d
