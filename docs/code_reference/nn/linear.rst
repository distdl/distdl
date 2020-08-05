============
Linear Layer
============

.. contents::
    :local:
    :depth: 2


Overview
========

The Distributed Linear (or affine) layer uses distributed primitive layers
to build a distributed version of the PyTorch ``Linear`` layer.  That is,
it implements

.. math::
   y = Wx + b

where the tensors :math:`x`, :math:`y`, :math:`W`, and :math:`b` are
partitioned over a number of workers.

For the purposes of this documentation, we will assume that an arbitrary
global input tensor :math:`{x}` is partitioned by :math:`P_x` and that another
partition :math:`P_y` exists.  Additionally, we will assume that the weight
tensor :math:`W` is partitioned by :math:`P_W`.  The bias :math:`b` is
implicitly partitioned.

Implementation
==============

For the construction of this layer, we assume that the fundamental unit of
work is driven by dense subtensors of :math:`W`.  Thus, the structure of the
partition :math:`P_W` drives the design.

The distributed linear layer is an application of distributed GEMM.  The
optimal implementation will be system and problem dependent.  The current
implementation is greedy from the perspective of the number of workers.

.. note::
   Other algorithms, which reduce the number of required workers can be built
   from similar primitives.  We are happy to implement those if they are
   suggested.

The current implementation stores the learnable weights and biases inside
local sequential Linear layer functions.  To avoid double counting the bias,
only a subset of these local sequential Linear layers has an active bias
vector.

Assumptions
-----------

* The global input tensor :math:`x` has shape :math:`n_{\text{batch}} \times
  n_{\text{features in}}`.
* The input partition :math:`P_x` has shape :math:`1 \times P_{\text{f_in}}`,
  where :math:`P_{\text{f_in}}` is the number of workers partitioning the
  feature dimension of :math:`x`.
* The global output tensor :math:`y` has shape :math:`n_{\text{batch}} \times
  n_{\text{features out}}`.
* The output partition :math:`P_y` has shape :math:`1 \times P_{\text{f_out}}`,
  where :math:`P_{\text{f_out}}` is the number of workers partitioning the
  feature dimension of :math:`y`.

.. note::
   PyTorch admits input tensors of shape :math:`n_{\text{batch}} \times \dots
   \times n_{\text{features in}}` and output tensors of shape
   :math:`n_{\text{batch}} \times \dots \times n_{\text{features out}}`.
   DistDL does not explicitly support intermediate dimensions at this time.

* The weight tensor :math:`W` has shape :math:`n_{\text{features_out}} \times
  n_{\text{features_in}}`.  This follows PyTorch.
* The weight partition :math:`P_W` has shape :math:`P_{\text{f_out}} \times
  P_{\text{f_in}}`.

.. note::
   The bias vectors are stored on the 0th *column* of :math:`P_w`.  Hence, it
   is implicitly partitioned by a factor of :math:`P_{\text{f_in}}`.
   Following PyTorch, if the bias is turned off, no subtensors have bias
   terms.

.. figure:: /_images/linear_example_01.png
   :alt: Example setup for distributed linear layer.

   An example setup for a distributed linear layer, where :math:`P_x` has
   shape :math:`1 \times 4`, :math:`P_y` has shape :math:`1 \times 3`, and
   :math:`P_W` has shape :math:`3 \times 4`.

Forward
-------

Under the above assumptions, the forward algorithm is:

1. Use a :ref:`code_reference/nn/broadcast:Broadcast Layer` to broadcast
   subtensors of :math:`x` from :math:`P_x` over the columns of :math:`P_W`.

.. figure:: /_images/linear_example_02.png
   :alt: Example forward broadcast in the distributed linear layer.

   Subtensors of :math:`x` are broadcast down the four columns of
   :math:`P_W`.

2. Perform the local forward linear layer application using a PyTorch Linear
   layer.  Note that the bias is only added on the 0th column of :math:`P_W`.
   Each worker now has a portion of the output vector :math:`y`.  In the rows
   of :math:`P_W` the results are partial contributions to the output feature
   degrees-of-freedom.

.. figure:: /_images/linear_example_03.png
   :alt: Example forward linear application in the distributed linear layer.

   Local application of linear layer.  Bias is present only in 0th column.

3. Use a :ref:`code_reference/nn/sum_reduce:SumReduce Layer` to reduce
   the subtensors of :math:`y` over the rows of :math:`P_W` into :math:`P_y`.
   Only one subtensor in each row of :math:`P_W` contains the a subtensor of
   the bias, so the output tensor correctly assimilates the bias.

   .. note::
      This sum-reduction requires one of the partitions to be transposed.

.. figure:: /_images/linear_example_04.png
   :alt: Example forward sum-reduction in the distributed linear layer.

   Subtensors of :math:`y` are assembled via sum-reduction along the three
   rows of :math:`P_W`.

Adjoint
-------

The adjoint algorithm is not explicitly implemented.  PyTorch's ``autograd``
feature automatically builds the adjoint of the Jacobian of the distributed
linear forward application.  Essentially, the algorithm is as follows:

1. Broadcast the subtensors of the gradient output, :math:`\delta y` from
   :math:`P_y` along the rows of :math:`P_W`.

.. figure:: /_images/linear_example_05.png
   :alt: Example adjoint sum-reduction in the distributed linear layer.

   Subtensors of :math:`\delta y` are broadcast across the three rows of
   :math:`P_W`.

2. Each worker in :math:`P_W` computes its local part of :math:`\delta W` and
   :math:`\delta x` using the PyTorch implementation of the adjoint of the
   Jacobian of the local sequential linear layer.  If the bias is required,
   the 0th column of :math:`P_W` also computes :math:`\delta b` similarly.

.. figure:: /_images/linear_example_06.png
   :alt: Example adjoint linear application in the distributed linear layer.

   Local computation of subtensors of :math:`\delta x`, :math:`\delta W`, and
   :math:`\delta b`.

3. Sum-reduce the subtensors of the gradient input, :math:`\delta x`, along
   the rows of :math:`P_W` into :math:`P_x`.

.. figure:: /_images/linear_example_07.png
   :alt: Example adjoint broadcast in the distributed linear layer.

   Subtensors of :math:`\delta x` are assembled via sum-reduction along the
   four columns of :math:`P_W`.


Examples
========

To apply a linear layer which maps a tensor on a ``1 x 4`` partition to a
tensor on a ``1 x 3`` partition:

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 4))
>>> P_x = P_x_base.create_cartesian_topology_partition([1, 4])
>>>
>>> P_y_base = P_world.create_partition_inclusive(np.arange(4, 7))
>>> P_y = P_y_base.create_cartesian_topology_partition([1, 3])
>>>
>>> P_W_base = P_world.create_partition_inclusive(np.arange(0, 12))
>>> P_W = P_W_base.create_cartesian_topology_partition([3, 4])
>>>
>>> in_features = 16
>>> out_features = 12
>>>
>>> x_local_shape = np.array([1, 4])
>>>
>>> layer = DistributedLinear(P_x, P_y, P_W, in_features, out_features)
>>>
>>> x = zero_volume_tensor()
>>> if P_x.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = layer(x)

API
===

.. currentmodule:: distdl.nn

.. autoclass:: DistributedLinear
    :members:
