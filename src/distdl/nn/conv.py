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

    def __new__(cls, P_x, P_y=None, P_w=None, *args, **kwargs):

        # If P_x is only partition specified
        if P_y is None and P_w is None:
            return cls.DistributedFeatureConvType(P_x, *args, **kwargs)

        if P_y is not None and P_w is not None:

            P_x_features = P_x.shape[2:]
            P_y_features = P_y.shape[2:]

            if np.all(P_x_features == 1) and np.all(P_y_features == 1):
                return cls.DistributedChannelConvType(P_x, P_y, P_w, *args, **kwargs)

            return cls.DistributedGeneralConvType(P_x, P_y, P_w, *args, **kwargs)

        raise ValueError("Cannot determine valid matching class.")


class DistributedConv1d(DistributedConvSelector):

    DistributedGeneralConvType = DistributedGeneralConv1d
    DistributedFeatureConvType = DistributedFeatureConv1d
    DistributedChannelConvType = DistributedChannelConv1d


class DistributedConv2d(DistributedConvSelector):

    DistributedGeneralConvType = DistributedGeneralConv2d
    DistributedFeatureConvType = DistributedFeatureConv2d
    DistributedChannelConvType = DistributedChannelConv2d


class DistributedConv3d(DistributedConvSelector):

    DistributedGeneralConvType = DistributedGeneralConv3d
    DistributedFeatureConvType = DistributedFeatureConv3d
    DistributedChannelConvType = DistributedChannelConv3d
