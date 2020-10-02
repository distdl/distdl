import numpy as np
import torch

from distdl.nn.broadcast import Broadcast
from distdl.nn.mixins.conv_mixin import ConvMixin
from distdl.nn.module import Module
from distdl.nn.sum_reduce import SumReduce
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import TensorStructure


class DistributedChannelConvBase(Module, ConvMixin):
    r"""A channel-space partitioned distributed convolutional layer.

    This class provides the user interface to a distributed convolutional
    layer, where the input (and output) tensors are partitioned in the
    channel-dimension only.

    The base unit of work is given by the weight tensor partition.  The
    pattern is similar to that of the :class:`DistributedLinear layer
    <distdl.nn.DistributedLinear>`.  This class requires the following of the
    tensor partitions:

    1. :math:`P_x` over input tensor :math:`x` has shape :math:`1 \times
       P_{\text{c_in}} \times 1 \times \dots \times 1`.
    2. :math:`P_y` over input tensor :math:`y` has shape :math:`1 \times
       P_{\text{c_out}} \times 1 \times \dots \times 1`.
    3. :math:`P_W` over weight tensor :math:`W` has shape
       :math:`P_{\text{c_out}} \times P_{\text{c_in}}  \times 1 \times \dots
       \times 1`.

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

        super(DistributedChannelConvBase, self).__init__()

        # P_x is 1    x P_ci x 1 x ... x 1
        self.P_x = P_x
        # P_y is 1    x P_co x 1 x ... x 1
        self.P_y = P_y
        # P_w is P_co x P_ci x 1 x ... x 1
        self.P_w = P_w

        # Even inactive workers need some partition union
        P_union = self._distdl_backend.Partition()

        if not (self.P_x.active or
                self.P_y.active or
                self.P_w.active):
            return

        # This guarantees that P_union rank 0 has the kernel size, stride,
        # padding, and dilation factors
        P_union_temp = P_w.create_partition_union(P_x)
        P_union = P_union_temp.create_partition_union(P_y)

        # Ensure that all workers have the full size and structure of P_w
        P_w_shape = None
        if P_union.rank == 0:
            P_w_shape = np.array(P_w.shape, dtype=np.int)
        P_w_shape = P_union.broadcast_data(P_w_shape, root=0)

        # Release the temporary resources
        P_union_temp.deactivate()
        P_union.deactivate()

        P_co = P_w_shape[0]
        P_ci = P_w_shape[1]
        P_channels = [P_co, P_ci]

        # Ensure that P_x and P_w are correctly aligned.  We also produce a
        # new P_x that is shaped like 1 x P_ci x 1 x ... x 1, to assist with
        # broadcasts.
        P_x_new_shape = []
        if self.P_x.active:
            if(np.any(P_x.shape[2:] != P_w_shape[2:])):
                raise ValueError("Spatial components of P_x and P_w must match.")
            if(np.any(P_x.shape[2:] != np.ones(len(P_x.shape[2:])))):
                raise ValueError("Spatial components of P_x must be 1 x ... x 1.")
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
        # new P_y that is shaped like P_co x 1 x 1 x ... x 1, to assist with
        # broadcasts.
        P_y_new_shape = []
        if self.P_y.active:
            if(np.any(P_y.shape[2:] != P_w_shape[2:])):
                raise ValueError("Spatial components of P_y and P_w must match.")
            if(np.any(P_y.shape[2:] != np.ones(len(P_y.shape[2:])))):
                raise ValueError("Spatial components of P_y must be 1 x ... x 1.")
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

        self.serial = False
        if self.P_w.size == 1:
            self.serial = True
            self.conv_layer = self.TorchConvType(*args, **kwargs)
            return

        # Flag if the global bias is set
        self.global_bias = bias

        # Flags if current worker stores (part of) the bias locally.
        self.stores_bias = False

        if self.P_w.active:

            # Let the P_co column store the bias if it is to be used
            self.stores_bias = self.global_bias and (self.P_w.index[1] == 0)

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
            local_kwargs["bias"] = self.stores_bias

            self.conv_layer = self.TorchConvType(*args, **local_kwargs)

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

        self.x_broadcast = Broadcast(self.P_x, self.P_w, preserve_batch=True)
        self.y_sum_reduce = SumReduce(self.P_w, self.P_y, preserve_batch=True)

    def _distdl_module_setup(self, input):
        r"""Distributed (channel) convolution module setup function.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        # No setup is needed if the worker is not doing anything for this
        # layer.
        if not (self.P_x.active or
                self.P_y.active or
                self.P_w.active):
            return

        if self.serial:
            return

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

        x = self.x_broadcast(input)

        if self.P_w.active:
            x = self.conv_layer(x)

        y = self.y_sum_reduce(x)

        return y


class DistributedChannelConv1d(DistributedChannelConvBase):
    r"""A channel-partitioned distributed 1d convolutional layer.

    """

    TorchConvType = torch.nn.Conv1d


class DistributedChannelConv2d(DistributedChannelConvBase):
    r"""A channel-partitioned distributed 2d convolutional layer.

    """

    TorchConvType = torch.nn.Conv2d


class DistributedChannelConv3d(DistributedChannelConvBase):
    r"""A channel-partitioned distributed 3d convolutional layer.

    """

    TorchConvType = torch.nn.Conv3d
