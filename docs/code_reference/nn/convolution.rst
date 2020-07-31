==================
Convolution Layers
==================

Overview
========

.. Hand-write some stuff here for now.  In the future, when there is a
   single interface class, this info will go there instead.

Simple Convolution
------------------

.. automodule:: distdl.nn.conv_feature

General Convolution
-------------------

.. automodule:: distdl.nn.conv_general

Convolution Mixin
-----------------

.. automodule:: distdl.nn.mixins.conv_mixin


API
===

.. currentmodule:: distdl.nn.conv_feature

.. autoclass:: DistributedConvBase
    :members:
    :exclude-members: forward

.. autoclass:: DistributedConv1d
    :members:
    :exclude-members: forward

.. autoclass:: DistributedConv2d
    :members:
    :exclude-members: forward

.. autoclass:: DistributedConv3d
    :members:
    :exclude-members: forward


.. currentmodule:: distdl.nn.conv_general

.. autoclass:: DistributedGeneralConvBase
    :members:
    :exclude-members: forward

.. autoclass:: DistributedGeneralConv1d
    :members:
    :exclude-members: forward

.. autoclass:: DistributedGeneralConv2d
    :members:
    :exclude-members: forward

.. autoclass:: DistributedGeneralConv3d
    :members:
    :exclude-members: forward


.. currentmodule:: distdl.nn.mixins

.. autoclass:: ConvMixin
    :members:
