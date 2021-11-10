==================
Convolution Layers
==================

.. contents::
    :local:
    :depth: 2

Overview
========

DistDL's The Distributed Convolutional layers use the distributed primitive
layers to build various distributed versions of PyTorch ``ConvXd`` layers.
That is, it implements

.. math::
   y = w*x + b

where :math:`*` is the convolution operator and the tensors :math:`x`,
:math:`y`, :math:`w`, and :math:`b` are partitioned over a number of workers.

For the purposes of this documentation, we will assume that an arbitrary
global input tensor :math:`{x}` is partitioned by :math:`P_x`.  Another
partition :math:`P_y`, may exist depending on implementation.  similarly, the
weight tensor :math:`w` may also have its own partition is partitioned by
:math:`P_w`.  The bias :math:`b` is implicitly partitioned depending on the
nature of :math:`P_w`.


Implementation
==============

The partitioning of the input and output tensors strongly impacts the
necessary operations to perform a distributed convolution.  Consequently,
DistDL has multiple implementations to satisfy some special cases and the
general case.


Public Interface
----------------

DistDL provides a public interface to the many distributed convolution
implementations that follows the same pattern as other public interfaces, such
as the :ref:`code_reference/nn/linear:Linear Layer` and keeping in line with
the PyTorch interface.  The ``distdl.nn.conv`` module provides the
:class:`distdl.nn.conv.DistributedConv1d`,
:class:`distdl.nn.conv.DistributedConv2d`, and
:class:`distdl.nn.conv.DistributedConv3d` types, which through use the class
:class:`distdl.nn.conv.DistributedConvSelector` to dispatch an appropriate
implementation, based on the structure of :math:`P_x`, :math:`P_y`, and
:math:`P_W`.

Current implementations include those for:

1. :ref:`code_reference/nn/convolution:Feature-distributed Convolution`
2. :ref:`code_reference/nn/convolution:Channel-distributed Convolution`
3. :ref:`code_reference/nn/convolution:Generalized Distributed Convolution`


Feature-distributed Convolution
-------------------------------

The simplest distributed convolution implementation, and the one that
generally requires the least workers, has input (and outout) tensors that are
distributed in feature-space only.  This is also, likely, the most common
use-case.

Construction of this layer is driven by the partitioning of the input tensor
:math:`x`, only.  Thus, the partition :math:`P_x` drives the algorithm design.
With a pure feature-space partition, the output partition will have the same
structure, so there is no need to specify it.  Also, with no partition in the
channel dimension, the learnable weight tensor is assumed to be small enough
that it can trivially be stored by one worker.

Assumptions
~~~~~~~~~~~

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
* The weight tensor :math:`w` will have shape :math:`n_{c_{\text{out}}}
  \times n_{c_{\text{in}}} \times k_{D-1} \times \cdots \times k_0`.
* The weight partition does not necessarily explicitly exist, but implicitly
  has shape :math:`1 \times 1 \times 1 \times \cdots \times 1`.
* Any learnable bias is stored on the same worker as the learnable weights.

.. figure:: /_images/conv_feature_example_01.png
   :alt: Example setup for feature-distributed convolutional layer.

   An example setup for a 1D distributed convolutional layer, where :math:`P_x` has
   shape :math:`1 \times 1 \times 4`, :math:`P_y` has the same shape, and
   :math:`P_W` has shape :math:`1 \times 1 \times 1`.

Forward
~~~~~~~

Under the above assumptions, the forward algorithm is:

1. Use a :ref:`code_reference/nn/broadcast:Broadcast Layer` to broadcast
   the learnable :math:`w` from a single worker in :math:`P_x` to all of
   :math:`P_x`.  If necessary, a different broadcast layer, also from a
   single worker in :math:`P_x` to all of :math:`P_x` broadcasts the
   learnable bias :math:`b`.

   The weight and bias tensors, post broadcast, are used by the local
   convolution.

.. figure:: /_images/conv_feature_example_02.png
   :alt: Example forward broadcast in the feature-distributed convolutional layer.

   :math:`w` and :math:`b` are broadcast to all workers in :math:`P_x`.

2. Perform the halo exchange on the subtensors of :math:`x`.  Here, :math:`x_j`
   must be padded to accept local halo regions (in a potentially unbalanced
   way) before the halos are exchanged.  The output of this operation is
   :math:`\hat x_j`.

.. figure:: /_images/conv_feature_example_03.png
   :alt: Example forward padding of subtensors of x in feature-distributed convolutional layer.

   Subtensors of :math:`x`, :math:`x_j` must be padded to accept the halo
   data.

.. figure:: /_images/conv_feature_example_04.png
   :alt: Example forward halo exchange on subtensors of x in feature-distributed convolutional layer.

   Forward halos are exchanged on :math:`P_x`, creating :math:`\hat x_j`.

3. Perform the local forward convolution application using a PyTorch
   ``ConvXd`` layer.  The bias is added everywhere, as each workers output
   will be part of the output tensor.

.. figure:: /_images/conv_feature_example_05.png
   :alt: Example forward convolution in the feature-distributed convolutional layer.

   The :math:`y_i` subtensors are computed using native PyTorch layers.


4. The subtensors in the inputs and outputs of DistDL layers should always be
   able to be reconstructed into precisely the same tensor a sequential
   application will produce.  Because padding is explicitly added to the input
   tensor to account for the padding specified for the convolution, the output
   of the local convolution, :math:`y_i`, should exactly match that of the
   sequential layer.

.. figure:: /_images/conv_feature_example_06.png
   :alt: Example forward result of the feature-distributed convolutional layer.

Adjoint
~~~~~~~

The adjoint algorithm is not explicitly implemented.  PyTorch's ``autograd``
feature automatically builds the adjoint of the Jacobian of the
feature-distributed convolution forward application.  Essentially, the
algorithm is as follows:

1. The gradient output :math:`\delta y_i` is already distributed across its partition,
   so the adjoint of the Jacobian of the local convolutional layer can be applied to it.

.. figure:: /_images/conv_feature_example_07.png
   :alt: Example adjoint starting case in the feature-distributed convolutional layer.

2. Each worker computes its local contribution to :math:`\delta w` and :math:`\delta x`,
   given by :math:`\delta w_j` and :math:`\delta x_j`, using PyTorch's native implementation of
   the adjoint of the Jacobian of the local sequential convolutional layer.
   If the bias is required, each worker computes its local contribution to
   :math:`\delta b_j`, :math:`\delta \hat b` similarly.

.. figure:: /_images/conv_feature_example_08.png
   :alt: Example adjoint convolution in the feature-distributed convolutional layer.

   Subtensors of :math:`\delta w_j`, :math:`\delta \hat x_j`, and :math:`\delta
   b_j` are computed using native PyTorch layers.

3. The adjoint of the halo exchange is applied to :math:`\delta \hat x`,
   which is then unpadded, producing the gradient input :math:`\delta x`.

.. figure:: /_images/conv_feature_example_09.png
   :alt: Example adjoint halo exchange on subtensors of dx in feature-distributed convolutional layer.

   Adjoint halos of :math:`\delta \hat x` are exchanged on the :math:`P_x`.

.. figure:: /_images/conv_feature_example_10.png
   :alt: Example adjoint padding (unpadding) of subtensors of delta x in feature-distributed convolutional layer.

   Subtensors :math:`\delta \hat x_j` must be unpadded to after the halo
   regions are cleared :to create, creating :math:`\delta x`.

4. Sum-reduce the partial weight gradients, :math:`\delta w_j`, to produce
   the total gradient :math:`\delta w` on the relevant worker in :math:`P_x`.

   If required, do the same thing to produce :math:`\delta b` from each
   worker's :math:`\delta b_j`.

.. figure:: /_images/conv_feature_example_11.png
   :alt: Example adjoint broadcast in the feature-distributed convolutional layer.

   :math:`\delta w` and :math:`\delta b` are constructed from a sum-reduction
   on all workers in :math:`P_x`.


Channel-distributed Convolution
-------------------------------

DistDL provides a distributed convolution layer that supports partitions in
the channel-dimension only.  This pattern may be useful when layers are narrow
in feature space.

For the construction of this layer, we assume that the fundamental unit of
work is driven by dense channels in :math:`w`.  Thus, the structure of the
partition :math:`P_w` drives the design.  This layer admits differences
between :math:`P_x` and :math:`P_y`, so all three partitions, including
:math:`P_w`, must be specified.  It is assumed that there is no partitioning
in the feature-space for the input and output tensors.

Assumptions
~~~~~~~~~~~

* The global input tensor :math:`x` has shape :math:`n_{\text{b}} \times
  n_{c_{\text{in}}} \times n_{D-1} \times \cdots \times n_0`.
* The input partition :math:`P_x` has shape :math:`1 \times P_{c_{\text{in}}} \times 1
  \times \cdots \times 1`, where :math:`P_{c_{\text{in}}}` is the number of workers
  partitioning the channel dimension of :math:`x`.
* The global output tensor :math:`y` will have shape :math:`n_{\text{b}}
  \times n_{c_{\text{out}}} \times m_{D-1} \times \cdots \times m_0`.
  The precise values of :math:`m_{D-1} \times \cdots \times m_0` are
  dependent on the input shape and the kernel parameters.
* The output partition :math:`P_y` has shape :math:`1 \times P_{c_{\text{out}}} \times 1
  \times \cdots \times 1`, where :math:`P_{c_{\text{out}}}` is the number of workers
  partitioning the channel dimension of :math:`y`.
* The global weight tensor :math:`w` will have shape :math:`n_{c_{\text{out}}}
  \times n_{c_{\text{in}}} \times k_{D-1} \times \cdots \times k_0`.
* The weight partition, which partitions the entire weight tensor, :math:`P_w`
  has shape :math:`P_{c_{\text{out}}} \times P_{c_{\text{in}}} \times 1 \times
  \cdots \times 1`.
* The learnable bias, if required, is stored on a :math:`P_{c_{\text{out}}} \times 1
  \times 1 \times \cdots \times 1` subset of the weight partition.

.. figure:: /_images/conv_channel_example_01.png
   :alt: Example setup for channel-distributed convolutional layer.

   An example setup for a 1, channel distributed convolutional layer, where
   :math:`P_x` has shape :math:`1 \times 4 \times 1`, :math:`P_y` has the
   shape :math:`1 \times 3 \times 1`, and :math:`P_w` has shape :math:`3
   \times 4 \times 1`.

Forward
~~~~~~~

Under the above assumptions, the forward algorithm is:

1. Use a :ref:`code_reference/nn/broadcast:Broadcast Layer` to broadcast
   subtensors of :math:`x` from :math:`P_x` along the :math:`P_{c_{\text{out}}}`
   dimension of :math:`P_w`, creating local copies of :math:`x_j`.

.. .. figure:: /_images/conv_channel_example_02.png
..    :alt: Example forward broadcast in the channel-distributed convolutional layer.

..    Copies of subtensors of :math:`x` are broadcast to :math:`P_w`.

2. Perform the local forward convolutional layer application using a PyTorch ConvXd
   layer.  Note that the bias is only added on the specified subset of :math:`P_w`.
   Each worker now has a portion of a subtensor, denoted :math:`y_i`, of
   the global output vector.

.. .. figure:: /_images/conv_channel_example_03.png
..    :alt: Example forward convolutional layer application in the channel-distributed convolutional layer.

..    Partial subtensors of :math:`y` are computed using native PyTorch layers.

3. Use a :ref:`code_reference/nn/sum_reduce:SumReduce Layer` to reduce
   the partial subtensors of :math:`y` along the :math:`P_{c_{\text{in}}}`
   dimension of :math:`P_w` into :math:`P_y`. Only one subtensor in each row
   of :math:`P_w` contains the a subtensor of the bias, so the output tensor
   correctly assimilates the bias.

.. .. figure:: /_images/conv_channel_example_04.png
..    :alt: Example forward sum-reduction in the channel-distributed convolutional layer.

..    Subtensors of :math:`y` are assembled via sum-reduction from :math:`P_w` to :math:`P_y`.

Adjoint
~~~~~~~

The adjoint algorithm is not explicitly implemented.  PyTorch's ``autograd``
feature automatically builds the adjoint of the Jacobian of the
channel-distributed convolution forward application.  Essentially, the
algorithm is as follows:

1. Broadcast the subtensors of the gradient output, :math:`\delta y` from
   :math:`P_y` along the :math:`P_{c_{\text{in}}}` dimension of :math:`P_w`,
   creating copies :math:`y_i`.

.. .. figure:: /_images/conv_channel_example_05.png
..    :alt: Example adjoint sum-reduction in the channel-distributed convolutional layer.

..    Subtensors of :math:`\delta y` are broadcast from :math:`P_y` to :math:`P_w`.

2. Each worker in :math:`P_w` computes its local subtensor of :math:`\delta w` and
   its contribution to the subtensors of
   :math:`\delta x` using the PyTorch implementation of the adjoint of the
   Jacobian of the local sequential convolutional layer.  If the bias is required,
   relevant workers compute the local subtensors of :math:`\delta b` similarly.

.. .. figure:: /_images/conv_channel_example_06.png
..    :alt: Example adjoint convolutional application in the channel-distributed convolutional layer.

..    Local computation of subtensors of :math:`\delta \hat x`, :math:`\delta W`,
..    and :math:`\delta b`.

3. Sum-reduce the partial subtensors of the gradient input, :math:`\delta x_j`, along
   the :math:`P_{c_{\text{out}}}` dimension of :math:`P_w` into :math:`P_x`.

.. .. figure:: /_images/conv_channel_example_07.png
..    :alt: Example adjoint broadcast in the channel-distributed convolutional layer.

..    Subtensors of :math:`\delta x` are assembled via sum-reduction from :math:`P_w`
..    to :math:`P_x`.


Generalized Distributed Convolution
-----------------------------------

DistDL provides a distributed convolution layer that supports partitioning in
both channel- and feature-dimensions.  This pattern is expensive.  Each of the
previous two algorithms can be derived from this algorithm.

For the construction of this layer, we assume that the fundamental unit of
work is driven by dense channels in :math:`w`.  Thus, the structure of the
partition :math:`P_w` drives the design.  This layer admits differences
between :math:`P_x` and :math:`P_y`, so all three partitions, including
:math:`P_w`, must be specified.  Any non-batch dimension is allowed to be
partitioned.

Assumptions
~~~~~~~~~~~

* The global input tensor :math:`x` has shape :math:`n_{\text{b}} \times
  n_{c_{\text{in}}} \times n_{D-1} \times \cdots \times n_0`.
* The input partition :math:`P_x` has shape :math:`1 \times P_{c_{\text{in}}} \times P_{D-1}
  \times \cdots \times P_0`, where :math:`P_{c_{\text{in}}}` is the number of workers
  partitioning the channel dimension of :math:`x` and :math:`P_{d}` is the number of workers
  partitioning the :math:`d^{\text{th}}` feature dimension of :math:`x`.
* The global output tensor :math:`y` will have shape :math:`n_{\text{b}}
  \times n_{c_{\text{out}}} \times m_{D-1} \times \cdots \times m_0`.
  The precise values of :math:`m_{D-1} \times \cdots \times m_0` are
  dependent on the input shape and the kernel parameters.
* The output partition :math:`P_y` has shape :math:`1 \times P_{c_{\text{out}}}
  \times P_{D-1} \times \cdots \times P_0`, where :math:`P_{c_{\text{out}}}` is
  the number of workers partitioning the channel dimension of :math:`y` and the
  feature partition is the same as :math:`P_x`.
* The weight tensor :math:`w` will have shape :math:`n_{c_{\text{out}}}
  \times n_{c_{\text{in}}} \times k_{D-1} \times \cdots \times k_0`.
* The weight partition :math:`P_w` has shape :math:`P_{c_{\text{out}}} \times P_{c_{\text{in}}}
  \times P_{D-1} \times \cdots \times P_0`.
* The learneable weights are stored on a :math:`P_{c_{\text{out}}} \times P_{c_{\text{in}}}
  \times 1 \times \cdots \times 1` subset of the weight partition.
* Any learnable bias is stored on a :math:`P_{c_{\text{out}}} \times 1
  \times 1 \times \cdots \times 1` subset of the weight partition.

.. .. figure:: /_images/conv_general_example_01.png
..    :alt: Example setup for general distributed convolutional layer.

..    An example setup for a 1D distributed convolutional layer, where
..    :math:`P_x` has shape :math:`1 \times 4 \times 4`, :math:`P_y` has the
..    shape :math:`1 \times 3 \times 4`, and :math:`P_w` has shape :math:`3
..    \times 4 \times 4`.

Forward
~~~~~~~

Under the above assumptions, the forward algorithm is:

1. Perform the halo exchange on the subtensors of :math:`x`.  Here, :math:`x_j`
   must be padded to accept local halo regions (in a potentially unbalanced
   way) before the halos are exchanged.  The output of this combined
   operations is :math:`\hat x_j`.

.. .. figure:: /_images/conv_general_example_02.png
..    :alt: Example forward padding of subtensors of x in general distributed convolutional layer.

..    :math:`x` must be padded to accept the halo data, creating :math:`\hat x`.

.. .. figure:: /_images/conv_general_example_03.png
..    :alt: Example forward halo exchange on subtensors of x in general distributed convolutional layer.

..    Forward halos are exchanged on :math:`P_x`.

2. Use a :ref:`code_reference/nn/broadcast:Broadcast Layer` to broadcast
   the local learnable subtensors of :math:`w` along the first two
   (:math:`P_{c_{\text{out}}} \times P_{c_{\text{in}}}`) dimensions of
   :math:`P_w` to all of :math:`P_w` (creating copies of :math:`w_{ij}`).  If
   necessary, a different broadcast layer broadcasts the local learnable
   subtensors of :math:`b` also from the first dimension
   (:math:`P_{c_{\text{out}}}`) of :math:`P_w` to the subset of :math:`P_w`
   which requires it (creating local copies of :math:`b_i`).

.. .. figure:: /_images/conv_general_example_04.png
..    :alt: Example forward broadcast in the general distributed convolutional layer.

..    Learnable subtensors of :math:`w` are broadcast within :math:`P_w`.

.. .. figure:: /_images/conv_general_example_05.png
..    :alt: Example forward broadcast in the general distributed convolutional layer.

..    Learnable subtensors of :math:`b` are broadcast within a subset of :math:`P_w`.

3. Use a :ref:`code_reference/nn/broadcast:Broadcast Layer` to broadcast
   :math:`\hat x` along the matching dimensions of :math:`P_w`.

.. .. figure:: /_images/conv_general_example_06.png
..    :alt: Example forward broadcast in the general distributed convolutional layer.

..    Subtensors of :math:`\hat x` are broadcast from :math:`P_x` to :math:`P_w`.

4. Perform the local forward convolution application using a PyTorch
   ``ConvXd`` layer.  The bias is added only in the subset of , as each workers output
   will be part of the output tensor.

.. .. figure:: /_images/conv_general_example_07.png
..    :alt: Example forward convolution in the general distributed convolutional layer.

..    Subtensors of :math:`\hat y` are computed using native PyTorch layers.

5. Use a :ref:`code_reference/nn/sum_reduce:SumReduce Layer` to reduce
   the subtensors of :math:`\hat y` along the matching dimensions of
   :math:`P_w`. Only one subtensor in each reduction dimension contains the a
   subtensor of the bias, so the output tensor correctly assimilates the bias.

.. .. figure:: /_images/conv_general_example_08.png
..    :alt: Example forward sum-reduction in the general distributed convolutional layer.

..    Subtensors of :math:`y` are assembled via sum-reduction along matching
..    dimensions of of :math:`P_w`.

6. The subtensors in the inputs and outputs of DistDL layers should always be
   able to be reconstructed into precisely the same tensor a sequential
   application will produce.  Because padding is explicitly added to the input
   tensor to account for the padding specified for the convolution, the output
   of the local convolution, :math:`y_i`, should exactly match that of the
   sequential layer..

.. .. figure:: /_images/conv_general_example_09.png
..    :alt: Example forward unpadding in the general distributed convolutional layer.

..    Any additional padding is removed, creating :math:`y`.

Adjoint
~~~~~~~

The adjoint algorithm is not explicitly implemented.  PyTorch's ``autograd``
feature automatically builds the adjoint of the Jacobian of the
channel-distributed convolution forward application.  Essentially, the
algorithm is as follows:

1. The gradient output :math:`\delta y_i` is already distributed across its partition,
   so the adjoint of the Jacobian of the local convolutional layer can be applied to it.

.. .. figure:: /_images/conv_general_example_10.png
..    :alt: Example adjoint unpadding in the feature-distributed convolutional layer.

..    Any required padding is added to subtensors of :math:`y`.

2. Broadcast the subtensors of the gradient output, :math:`\delta y_i` from
   :math:`P_y` along the matching dimensions of :math:`P_w`.

.. .. figure:: /_images/conv_general_example_11.png
..    :alt: Example adjoint sum-reduction in the channel-distributed convolutional layer.

..    Subtensors of :math:`\delta y` are broadcast from :math:`P_y` to :math:`P_w`.

3. Each worker in :math:`P_w` computes its local part of :math:`\delta w_{ij}` and
   :math:`\delta x_j` using the PyTorch implementation of the adjoint of the
   Jacobian of the local sequential convolutional layer.  If the bias is
   required, the relevant workers in :math:`P_w` also compute their portion of
   :math:`\delta b_i` similarly.

.. .. figure:: /_images/conv_general_example_12.png
..    :alt: Example adjoint convolutional application in the channel-distributed convolutional layer.

..    Local computation of subtensors of :math:`\delta x`, :math:`\delta w`, and
..    :math:`\delta b`.

4. Sum-reduce the local contributions of :math:`\delta x_j` along the matching
   dimensions of :math:`P_w` to :math:`P_x`.

.. .. figure:: /_images/conv_general_example_13.png
..    :alt: Example forward broadcast in the general distributed convolutional layer.

..    Subtensors of :math:`\delta \hat x` are broadcast within :math:`P_w`.

5. Sum-reduce the local partial weight gradients along the first two
   (:math:`P_{c_{\text{out}}} \times P_{c_{\text{in}}}`) dimensions of
   :math:`P_w`, to produce total subtensors of :math:`\delta w`.

   If required, do the same thing to produce local subtensors of :math:`\delta
   b` from each relevant worker's local subtensor of :math:`\delta b`.

.. .. figure:: /_images/conv_general_example_14.png
..   :alt: Example adjoint broadcast in the feature-distributed convolutional layer.

..   :math:`\delta w` and :math:`\delta b` are constructed from a sum-reduction
..   involving relevant workers.

6. The adjoint of the halo exchange is applied to :math:`\delta \hat x_j`,
   which is then unpadded, producing the gradient input :math:`\delta x_j`.

.. .. figure:: /_images/conv_general_example_15.png
..   :alt: Example adjoint halo exchange on subtensors of dx in feature-distributed convolutional layer.

..   Adjoint halos are exchanged on :math:`\delta \hat x`.

.. .. figure:: /_images/conv_general_example_16.png
..   :alt: Example adjoint padding (unpadding) of subtensors of delta x in feature-distributed convolutional layer.

..   :math:`\delta \hat x` must be unpadded to after the halo regions are cleared
..   :to create, creating :math:`\delta x`.

Convolution Mixin
-----------------

Some distributed convolution layers require more than their local subtensor to
compute the correct local output.  This is governed by the "left" and "right"
extent of the convolution kernel.  As these calculations are the same for all
convolutions, they are mixed in to every convolution layer requiring a halo
exchange.

Assumptions
~~~~~~~~~~~

* Convolution kernels are centered.
* When a kernel has even size, the left side of the kernel is the shorter side.

.. warning::
   Current calculations of the subtensor index ranges required do not correctly
   take stride and dilation into account.


Examples
========


API
===

``distdl.nn.conv``
------------------

.. currentmodule:: distdl.nn.conv

.. autoclass:: DistributedConv1d
    :members:

.. autoclass:: DistributedConv2d
    :members:

.. autoclass:: DistributedConv3d
    :members:

.. autoclass:: DistributedConvSelector
    :members:


``distdl.nn.conv_feature``
--------------------------

.. currentmodule:: distdl.nn.conv_feature

.. autoclass:: DistributedFeatureConvBase
    :members:

.. autoclass:: DistributedFeatureConv1d
    :members:

.. autoclass:: DistributedFeatureConv2d
    :members:

.. autoclass:: DistributedFeatureConv3d
    :members:


``distdl.nn.conv_channel``
--------------------------

.. currentmodule:: distdl.nn.conv_channel

.. autoclass:: DistributedChannelConvBase
    :members:

.. autoclass:: DistributedChannelConv1d
    :members:

.. autoclass:: DistributedChannelConv2d
    :members:

.. autoclass:: DistributedChannelConv3d
    :members:


``distdl.nn.conv_general``
--------------------------

.. currentmodule:: distdl.nn.conv_general

.. autoclass:: DistributedGeneralConvBase
    :members:

.. autoclass:: DistributedGeneralConv1d
    :members:

.. autoclass:: DistributedGeneralConv2d
    :members:

.. autoclass:: DistributedGeneralConv3d
    :members:


.. currentmodule:: distdl.nn.mixins

.. autoclass:: ConvMixin
    :members:
    :private-members:
