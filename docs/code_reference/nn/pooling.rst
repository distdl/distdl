==============
Pooling Layers
==============

.. contents::
    :local:
    :depth: 2

Overview
========

DistDL's The Distributed Pooling layers use the distributed primitive
layers to build various distributed versions of PyTorch Pooling layers.

For the purposes of this documentation, we will assume that an arbitrary
global input tensor :math:`{x}` is partitioned by :math:`P_x`.


Implementation
==============

Currently, all pooling operations follow the same pattern.  Therefore,
a single base class implements the core distributed work and the
actual pooling operation is deferred to the underlying PyTorch layer.

As there are no learnable parameters in these layers, the parallelism is
induced by partitions of the inout (and therefore output) tensors. Here, input
(and outout) tensors that are distributed in feature-space only.

Construction of this layer is driven by the partitioning of the input tensor
:math:`x`, only.  Thus, the partition :math:`P_x` drives the algorithm design.
With a pure feature-space partition, the output partition will have the same
structure, so there is no need to specify it.

In general, due to the non-centered nature of pooling kernels, halos will
be one-sided.  See the `motivating paper <https://arxiv.org/abs/2006.03108>`_
for more details.

Assumptions
-----------

* The global input tensor :math:`x` has shape :math:`n_{\text{b}} \times
  n_{c_{\text{in}}} \times n_{D-1} \times \cdots \times n_0`.
* The input partition :math:`P_x` has shape :math:`1 \times 1 \times P_{D-1}
  \times \cdots \times P_0`, where :math:`P_{d}` is the number of workers
  partitioning the :math:`d^{\text{th}}` feature dimension of :math:`x`.
* The global output tensor :math:`y` will have shape :math:`n_{\text{b}}
  \times n_{c_{\text{out}}} \times m_{D-1} \times \cdots \times m_0`.
  The precise values of :math:`m_{D-1} \times \cdots \times m_0` are
  dependent on the input shape and the kernel parameters.
* The output partition :math:`P_y` implicitly has the same shape as
  :math:`P_x`.


Forward
~~~~~~~

Under the above assumptions, the forward algorithm is:

1. Perform the halo exchange on the subtensors of :math:`x`.  Here, :math:`x_j`
   must be padded to accept local halo regions (in a potentially unbalanced
   way) before the halos are exchanged.  The output of this operation is
   :math:`\hat x_j`.

2. Perform the local forward pooling application using a PyTorch
   pooling layer.  The bias is added everywhere, as each workers output
   will be part of the output tensor.


Adjoint
~~~~~~~

The adjoint algorithm is not explicitly implemented.  PyTorch's ``autograd``
feature automatically builds the adjoint of the Jacobian of the
feature-distributed convolution forward application.  Essentially, the
algorithm is as follows:


1. Each worker computes its local contribution to :math:`\delta x`,
   given by :math:`\delta x_j`, using PyTorch's native implementation of
   the adjoint of the Jacobian of the local sequential pooling layer.

2. The adjoint of the halo exchange is applied to :math:`\delta \hat x`,
   which is then unpadded, producing the gradient input :math:`\delta x`.

Pooling Mixin
-------------

Some distributed pooling layers require more than their local subtensor to
compute the correct local output.  This is governed by the "left" and "right"
extent of the pooling window.  As these calculations are the same for all
pooling operations, they are mixed in to every pooling layer requiring a halo
exchange.

Assumptions
~~~~~~~~~~~

* Pooling kernels are not centered, the origin of the window is the "upper left"
  entry.
* When a kernel has even size, the left side of the kernel is the shorter side.

.. warning::
   Current calculations of the subtensor index ranges required do not correctly
   take padding and dilation into account.


Examples
========

API
===


``distdl.nn.pooling``
--------------------------

.. currentmodule:: distdl.nn.pooling

.. autoclass:: DistributedPoolingBase
    :members:

.. autoclass:: DistributedAvgPool1d
    :members:

.. autoclass:: DistributedAvgPool2d
    :members:

.. autoclass:: DistributedAvgPool3d
    :members:

.. autoclass:: DistributedMaxPool1d
    :members:

.. autoclass:: DistributedMaxPool2d
    :members:

.. autoclass:: DistributedMaxPool3d
    :members:

.. currentmodule:: distdl.nn.mixins

.. autoclass:: PoolingMixin
    :members:
    :private-members:
