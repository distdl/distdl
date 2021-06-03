==================
AllSumReduce Layer
==================

.. contents::
    :local:
    :depth: 2

Overview
========

The AllSumReduce distributed data movement primitive sums data on a within a
set of workers in a Partition.

In DistDL, all-sum-reductions sum data from (sub)tensors along slices of a
partition.  The all-sum-reduce operation applies for partitions with and
without a (Cartesian) topology.

For the purposes of this documentation, we will assume that an arbitrary
global input tensor :math:`{x}` is partitioned by :math:`P_x`.

.. note::
   The definition of a all-sum-reduction in DistDL goes beyond the classical
   parallel reduction operation, for example, ``MPI_Allreduce()`` in MPI.  Such
   reductions typically assume 1-dimensional arrays, reduced *within* a group
   of workers, and neither impose nor exploit topological structure on the set
   of workers.

Motivation
==========

In distributed deep learning, there are many applications of the all-reduction
primitive.  For example, in normalization layers, including
distributed :ref:`code_reference/nn/batchnorm:Batch Normalization Layers`,
global tensor statistics are required on all workers.


Implementation
==============

A back-end functional implementation supporting DistDL
:class:`~distdl.nn.AllSumReduce` allows users to specify which dimensions
of the partition the reductions happen along.  No other options are
required because the all-reduction occurs within the input partition.

Input tensors may be partitioned by this partition but they are not required
to be.  If the AllSumReduce is part of a broader all-sum-reduction on a
tensor (along specific dimesions) then the local reduction must be performed
first and the distributed reduction afterward.  (This is normal for
distributed all-reductions.)

Assumptions
-----------

* The all-sum-reduction operation is *not* in-place.  Even if the operation is
  equivalent to an identity (no dimensions are used in the reduction), a
  Torch ``.clone()`` of the tensor is returned.

Forward
-------

The forward operation sums subtensors within :math:`P_x`, within subpartitions
of the input partition, as specified by the user, and broadcasts
the results within the same subpartition.  The reduce and broadcast operations
are not necessarily explicit.

* A worker that is active in :math:`P_x` will take a subtensor
  of :math:`x` as input and return a subtensor of :math:`y` as output.
* A worker that is not active in :math:`P_x` will take a zero-volume tensor
  as input and return a clone of that tensor as output.

This class provides only an interface to the back-end implementation of the
forward algorithm.  This interface does not impose any mechanism for
performing the reduction.  Performance details and optimizations are back-end
dependent.

The back-end forward operation is implemented through the `PyTorch autograd
<https://pytorch.org/docs/stable/autograd.html>`_ functional interface and
called through the AllSumReduce :math:`~distdl.nn.AllSumReduce.forward` function.

Adjoint
-------

AllSumReduce is self-adjoint.  Thus, the adjoint (backward) operation is exactly
the same as the forward operation.

This class provides only an interface to the back-end implementation of the
adjoint algorithm.  This interface does not impose any mechanism for
performing this broadcast.  Performance details and optimizations are
back-end dependent.

The adjoint operation (PyTorch grad function class) is generated automatically
via autograd and calls the ``backward()`` function implemented by the back-end
functional interface.

Examples
========

To reduce a 2-dimensional tensor that lives on a ``2 x 2 x 3`` partition
along the last two dimesions:

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 12))
>>> P_x = P_x_base.create_cartesian_topology_partition([2, 2, 3])
>>>
>>> x_local_shape = np.array([7, 5])
>>>
>>> reduce_dims = (1, 2)
>>> layer = AllSumReduce(P_x, reduce_dims)
>>>
>>> x = zero_volume_tensor()
>>> if P_x.active:
>>>     x = torch.rand(*x_local_shape)
>>>
>>> y = layer(x)

Here, each subtensor of :math:`{y}` is the sum of the subtensors of
:math:`{x}` from 6 workers.

API
===

.. currentmodule:: distdl.nn

.. autoclass:: AllSumReduce
    :members:

