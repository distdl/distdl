import numpy as np

from distdl.nn.conv_channel import DistributedChannelConv1d
from distdl.nn.conv_channel import DistributedChannelConv2d
from distdl.nn.conv_channel import DistributedChannelConv3d
from distdl.nn.conv_feature import DistributedFeatureConv1d
from distdl.nn.conv_feature import DistributedFeatureConv2d
from distdl.nn.conv_feature import DistributedFeatureConv3d
from distdl.nn.conv_general import DistributedGeneralConv1d
from distdl.nn.conv_general import DistributedGeneralConv2d
from distdl.nn.conv_general import DistributedGeneralConv3d


class DistributedConvSelector:
    r""" Base class for automatic distributed convolution layer selectors.

    Uses information from tensor partition inputs to dispatch creation of
    distributed convolutional layers to the best candidate class.

    Subclasses must specify the following types:
    1. An implementation supporting channel-only partitioning.
    2. An implementation supporting feature-only partitioning.
    3. An implemenation supporting any (general) partitioning.

    Parameters
    ----------
    P_x :
        Partition of input tensor.
    P_y : optional
        Partition of output tensor.
    P_w : optional
        Partition of the weight tensor.

    """

    def __new__(cls, P_x, P_y=None, P_w=None, *args, **kwargs):

        # If P_x is only partition specified, we assume a simple feature-only
        # partitioning scheme.
        if P_y is None and P_w is None:
            return cls.DistributedFeatureConvType(P_x, *args, **kwargs)

        # P_y and P_w are required for channel and general convolutions
        if P_y is not None and P_w is not None:

            P_x_features = P_x.shape[2:]
            P_y_features = P_y.shape[2:]

            # If there is no feature-space partitioning, then all feature
            # partitions will have size 1.  Then, channel only partitioning
            # is assumed.
            if np.all(P_x_features == 1) and np.all(P_y_features == 1):
                return cls.DistributedChannelConvType(P_x, P_y, P_w,
                                                      *args, **kwargs)

            # In all other cases, the generalized type is appropriate
            return cls.DistributedGeneralConvType(P_x, P_y, P_w,
                                                  *args, **kwargs)

        raise ValueError("Cannot determine valid matching class.")


class DistributedConv1d(DistributedConvSelector):
    r""" Public interface for 1D Distributed Convolutions.
    """

    DistributedChannelConvType = DistributedChannelConv1d
    DistributedFeatureConvType = DistributedFeatureConv1d
    DistributedGeneralConvType = DistributedGeneralConv1d


class DistributedConv2d(DistributedConvSelector):
    r""" Public interface for 2D Distributed Convolutions.
    """

    DistributedChannelConvType = DistributedChannelConv2d
    DistributedFeatureConvType = DistributedFeatureConv2d
    DistributedGeneralConvType = DistributedGeneralConv2d


class DistributedConv3d(DistributedConvSelector):
    r""" Public interface for 3D Distributed Convolutions.
    """

    DistributedChannelConvType = DistributedChannelConv3d
    DistributedFeatureConvType = DistributedFeatureConv3d
    DistributedGeneralConvType = DistributedGeneralConv3d
