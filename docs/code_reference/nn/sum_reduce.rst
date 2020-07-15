===============
SumReduce Layer
===============

.. contents::
    :local:
    :depth: 2

Overview
========

The SumReduce distributed data movement primitive sums data on a set of workers
to one worker (or set of workers).

In DistDL, sum-reductions sum data from subtensors on one partition to another
partition.  The sum-reduce operation applies for partitions with and without a
(Cartesian) topology.  Topologies may be mixed if the requirements supporting
the :ref:`code_reference/nn/sum_reduce:Reduction Rules` are satisfied.

For the purposes of this documentation, we will assume that an arbitrary
global input tensor :math:`{x}` is partitioned by :math:`P_x` and that another
partition :math:`P_y` exists.

.. note::
   The definition of a sum-reduction in DistDL goes beyond the classical
   parallel reduction operation, for example, ``MPI_Reduce()`` in MPI.  Such
   reductions typically assume 1-dimensional arrays, reduced *within* a group
   of workers, and neither impose nor exploit topological structure on the set
   of workers.

Motivation
==========

In distributed deep learning, there are many applications of the reduction
primitive.  Depending on computation distribution, and thus partition
structure, any tensor in a distributed layer may need to be reduced.  For
example, in distributed :ref:`code_reference/nn/convolution:Convolution
Layers`, a simple partition of the input tensor in the channel dimension
requires that outputs, which are partially computed on a number of workers, be
sum-reduced before returning the final tensors. In distributed :ref:`Linear
Layers <code_reference/nn/linear:Linear Layer>`, the weight tensor is
partitioned and the output tensor needs to be reduced along the flattened
feature dimension.


Implementation
==============

A back-end functional implementation supporting DistDL
:class:`~distdl.nn.SumReduce` must follow the
:ref:`code_reference/nn/sum_reduce:Reduction Rules` and must also support the
following options:

* ``transpose_src``, a boolean which tells the reduction algorithm to
  transpose :math:`P_x` by implicitly reversing its shape. (Default ``False``)
* ``transpose_dest``, a boolean which tells the reduction algorithm to
  transpose :math:`P_y` by implicitly reversing its shape. (Default ``False``)
* ``preserve_batch``, a boolean which tells the reduction algorithm to ensure
  that any zero-volume outputs retain the batch size, or first dimension shape,
  of the input tensor.  (Default ``True``)

.. note::
   While it may appear that ``transpose_src`` and ``transpose_dest`` will have
   equivalent impact, this is not true because we allow :math:`\text{dim}(P_x)
   > \text{dim}(P_y)`.  Moreover, when it is true that they are equivalent,
   there can be semantic information conveyed by the choice of which partition
   to transpose.

To support some slightly different reduction patterns, users can implicitly
transpose the input or output partitions.  In either transposition, the
shape of the partition is implicitly reversed but the structure of the input
or output tensor is not changed.

Assumptions
-----------

* The sum-reduction operation is *not* in-place.  Even if a worker is both a
  data source and a data destination and even if that data is the same (i.e.,
  the operation is locally an identity), a Torch ``.clone()`` of the tensor is
  returned.
* A worker may be active in the input partition, output partition, both
  partitions, or neither partition.
* If a worker is active in both partitions, its own input may not contribute
  to its output.

Forward
-------

The forward operation sums subtensors from :math:`P_x` to :math:`P_y`, along
the reduceable dimensions of the input partition.

* A worker that is active in :math:`P_x` and :math:`P_y` will take a subtensor
  of :math:`x` as input and return a subtensor of :math:`y` as output.
* A worker that is active only in :math:`P_x` will take a subtensor of
  :math:`x` as input and return a zero-volume tensor, optionally with the same
  first dimension (batch size) as the input.
* A worker that is active only in :math:`P_y` will take a zero-volume tensor
  as input and return a subtensor of :math:`y` as output.
* A worker that is active in neither partition will take a zero-volume tensor
  as input and return a clone of that tensor as output.

This class provides only an interface to the back-end implementation of the
forward algorithm.  This interface does not impose any mechanism for
performing the reduction.  Performance details and optimizations are back-end
dependent.

The back-end forward operation is implemented through the `PyTorch autograd
<https://pytorch.org/docs/stable/autograd.html>`_ functional interface and
called through the SumReduce :meth:`~distdl.nn.SumReduce.forward` function.

Adjoint
-------

The adjoint (backward) operation broadcasts tensors from :math:`P_y` back to
:math:`P_x`, along the reduceable dimensions of the input partition.

* A worker that is active in :math:`P_x` and :math:`P_y` will take a subtensor
  of :math:`\partial y` as input and return a (potentially different)
  subtensor of :math:`\partial x` as output.
* A worker that is active only in :math:`P_x` will take a zero-volume tensor
  as input and return a subtensor of :math:`\partial x` as output.
* A worker that is active only in :math:`P_y` will take a subtensor of
  :math:`\partial y` as input and return a zero-volume tensor, optionally with
  the same first dimension (batch size) as the original tensor :math:`x`.
* A worker that is active in neither partition will take a zero-volume tensor
  as input and return a clone of that tensor as output.

This class provides only an interface to the back-end implementation of the
adjoint algorithm.  This interface does not impose any mechanism for
performing this broadcast.  Performance details and optimizations are
back-end dependent.

The adjoint operation (PyTorch grad function class) is generated automatically
via autograd and calls the ``backward()`` function implemented by the back-end
functional interface.


Reduction Rules
===============

DistDL :class:`~distdl.nn.SumReduce` layers reduce along partition dimensions
following rules similar to the `NumPy broadcast rules
<https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules>`_.
Tensors mapped onto DistDL partitions will sum-reduce along a dimension when

1. the shape of :math:`P_x` in that dimension matches the shape of :math:`P_y`
   in that dimension, or
2. the shape of :math:`P_y` in that dimension is 1.

The major difference between these reduction rules and the NumPy broadcast
rules are that Rule 2 for NumPy arrays allows the either array to have shape 1
in that dimension, but in DistDL, only the output partition can have shape 1.
Like NumPy, :math:`P_x` and :math:`P_y` do not have to have the same shape,
however, it is required that :math:`\text{dim}(P_x) \ge \text{dim}(P_y)`.
When the dimensions do not match, :math:`P_y` is implicitly extended *to the
left* with ones until the dimension does match.

In the following examples, subtensors are defined by the black partition
borders.  There is one worker per subtensor in each partition.  Like colors
indicate subtensors that interact in the reduction.  For example, blue
subtensors in :math:`P_x` reduce to the blue subtensor in :math:`P_y`.

Standard Sum-reductions
-----------------------

Example 1
~~~~~~~~~

A partition with shape :math:`4` reduces to a partition with shape
:math:`1`.  The shape of the tensor is irrelevant.

.. figure:: /_images/sum_reduce_4_to_1.png
    :alt: Image of 4 to 1 reduction.

Example 2
~~~~~~~~~

A partition with shape :math:`2 \times 3` reduces to a partition with shape
:math:`1`. :math:`P_y` is implicitly extended to :math:`1 \times 1`.

.. figure:: /_images/sum_reduce_2x3_to_1.png
    :alt: Image of 2x3 to 1 reduction.

Example 3
~~~~~~~~~

A partition with shape :math:`3 \times 4` reduces to a partition with shape
:math:`3 \times 1`.

.. figure:: /_images/sum_reduce_3x4_to_3x1.png
    :alt: Image of 3x4 to 3x1 reduction.

Example 4
~~~~~~~~~

A partition with shape :math:`4 \times 4 \times 3` reduces to a partition
with shape :math:`1 \times 1 \times 3`.

.. figure:: /_images/sum_reduce_4x4x3_to_1x1x3.png
    :alt: Image of 4x4x3 to 1x1x3 reduction.

Example 5
~~~~~~~~~

A partition with shape :math:`3 \times 3 \times 2` **does not reduce** to a
partition with shape :math:`1 \times 1 \times 3`.

.. figure:: /_images/sum_reduce_3x3x2_to_1x1x3.png
    :alt: Image of failed 3x3x2 to 1x1x3 reduction.

Example 6
~~~~~~~~~

A partition with shape :math:`1 \times 3` **does not reduce** to a
partition with shape :math:`3 \times 1`.  If either ``transpose_src`` or
``transpose_dest`` are ``True``, then this *will* reduce.  However, it will
not be an identity reduction because there is no guarantee that no
data movement is required.

.. figure:: /_images/sum_reduce_1x3_to_3x1.png
    :alt: Image of failed 1x3 to 3x1 reduce.


Reductions with ``transpose_src = True``
----------------------------------------

When ``transpose_src`` is ``True``, the shape of the input partition is
implicitly reversed.  Thus, if :math:`P_x` has shape :math:`4 \times 3 \times
1`, the reduction behaves as if it is has shape :math:`1 \times 3 \times 4`.

Example 7
~~~~~~~~~

A partition with shape :math:`3 \times 4` reduces to a partition with shape
:math:`1 \times 3` if ``transpose_src = True``.

.. figure:: /_images/sum_reduce_3x4_to_1x3.png
    :alt: Image of 3x4 to 1x3 reduction using transpose_src.


Reductions with ``transpose_dest = True``
-----------------------------------------

When ``transpose_dest`` is ``True``, the shape of the output partition is
implicitly reversed.  Thus, if :math:`P_y` has shape :math:`1 \times 3 \times
4`, the reduction behaves as if it is has shape :math:`4 \times 3 \times 1`.

.. note::
    If :math:`P_y` has shape :math:`3 \times 4`, the transpose occurs *before*
    the left side is padded with ones, so the effective shape is :math:`1 \times 4
    \times 3`.

Example 8
~~~~~~~~~

A partition with shape :math:`3 \times 4` reduces to a partition with shape
:math:`4 \times 1` if ``transpose_dest = True``.

.. figure:: /_images/sum_reduce_3x4_to_4x1.png
    :alt: Image of 3x4 to 4x1 reduction using transpose_dest.

Examples
========

To reduce a 2-dimensional tensor that is partitioned by a ``4 x 3`` partition
onto a ``1 x 3`` partition:

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 12))
>>> P_x = P_x_base.create_cartesian_topology_partition([4, 3])
>>>
>>> P_y_base = P_world.create_partition_inclusive(np.arange(0, 3))
>>> P_y = P_y_base.create_cartesian_topology_partition([1, 3])
>>>
>>> x_local_shape = np.array([7, 5])
>>>
>>> layer = SumReduce(P_x, P_y, preserve_batch=False)
>>>
>>> x = zero_volume_tensor()
>>> if P_x.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = layer(x)

Here, each subtensor of :math:`{y}` is the sum of 4 subtensors of :math:`{x}`.

API
===

.. currentmodule:: distdl.nn

.. autoclass:: SumReduce
    :members:

