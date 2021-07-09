==============
Loss Functions
==============

.. contents::
    :local:
    :depth: 2


Overview
========

DistDL provides distributed implementations of many PyTorch loss functions.

For the purposes of this documentation, we will assume that arbitrary global
input and target tensors :math:`{x}` and :math:`{y}` are partitioned
by :math:`P_x`.

Implementation
==============

DistDL distributed loss functions are essentially wrappers around their
corresponding PyTorch loss functions, with the reductions computed properly
for parallel environments.  When reduced, the loss value is on a single worker.

For `reduction="none"`, as with the PyTorch losses, no reduction is performed
and each worker has its component of the element-wise loss.

When `reduction` is ``"sum"`` or ``"mean"`` or (``"batchmean"``), each worker
computes a local sum (the `reduction` mode for the base PyTorch layer is
``"sum"``) and the appropriate normalization factor (:math:`1`, the total
features, or the batch size) is applied *after* a DistDL SumReduce layer is
used to reduce the loss to the root worker.

The root partition is assembled when the distributed loss is instantiated and
consists, always, of the :math:`0^{\text{th}}` worker.  After the call,
the :math:`0^{\text{th}}` worker in :math:`P_x` has the true loss and all
other workers have invalid values.

PyTorch requires loss functions to be scalars (wrapped in a ``Tensor``) for the
``backward()`` method to work.  In
DistDL, :ref:`code_reference/nn/sum_reduce:SumReduce Layer` layers return
zero-volume tensors for workers that are not in the output partition. To
prevent optimization loops from needing to branch on the :math:`0^{\text
{th}}` worker to call ``backward()``, distributed losses use the
``ZeroVolumeCorrectorFunction()`` to convert zero-volume outputs to
meaningless scalar tensors in a ``forward()`` call and to convert any grad
input back to zero-volume tensors during the ``backward()`` phase.

.. note::
   DistDL distributed loss functions follow DistDL's design principles: the 
   communication is part of the mathematical formulation of the distributed
   network.  Thus, we do not all-reduce the result.  Only one worker has the
   true loss.

   However, our approach is *equivalent* to those that do perform the
   all-reduce. If an all-reduce is applied and the result is normalized,
   technically, nothing needs to be done in the adjoint phase.  The adjoint
   would be another normalized all-reduction, which is essentially the
   identity.

   Here, the forward operation includes only a sum-reduction, which induces
   a broadcast in the adjoint operation.  This sum-reduction followed by a
   broadcast is *precisely* an all-reduction.  However, it is induced
   naturally rather than imposed externally.

Assumptions
-----------

* The global input tensor :math:`x` has shape :math:`n_{\text{batch}} \times
  n_{D-1} \times \cdots \times n_0`, where :math:`D` is number of channel
  and feature dimensions.
* The global target tensor :math:`y` has the same shape as :math:`x` and is
  distributed such that the local shapes also match.
* The input partition :math:`P_x` has shape :math:`P_{\text{b}} \times P_{D-1}
  \times \cdots \times P_0`, where :math:`P_{d}` is the number of workers
  partitioning the :math:`d^{\text{th}}` channel or feature dimension of
  :math:`x` and :math:`y`.
* The worker with rank 0 returns the global loss, which has the same value as
  if it were computed sequentially.  All other workers return a scalar with
  value :math:`0.0`.

Forward
-------

Under the above assumptions, the forward algorithm is:

1. Compute the local loss.  If the reduction mode is ``"none"``, return the
   result of the PyTorch layer on the local input and target.  If another
   reduction is specified, apply the ``"sum"`` reduction mode to the local
   layer.

2. Use a :ref:`code_reference/nn/sum_reduce:SumReduce Layer` to reduce
   the local losses to the root worker.

3. On the :math:`0^{\text{th}}` worker, apply the correct normalization based
   on the `reduction` mode.

.. note:: 
   The normalization constant is computed in a pre-forward hook so that
   it can be re-used without more collective communication.

4. For the :math:`0^{\text{th}}` worker, return the global loss.  For all
   other workers in :math:`P_x`, return a scalar with value :math:`0.0`.

Adjoint
-------

The adjoint algorithm is not explicitly implemented.  PyTorch's ``autograd``
feature automatically builds the adjoint of the Jacobian of the distributed
loss calculation.  Essentially, the algorithm is as follows:

1. For the :math:`0^{\text{th}}` worker, the gradient output (input to
   ``backward()``) is preserved. For all other workers, convert that input to a
   zero-volume tensor.

2. On the :math:`0^{\text{th}}` worker, apply the adjoint of the normalization.

3. Broadcast the gradient output to all workers in :math:`P_x`.  This is the
   adjoint of the forward sum-reduce.

4. Compute the local adjoint application.

Examples
========

To apply a distributed loss layer on tensors mapped to a ``1 x 4`` partition:

>>> P_x_base = P_world.create_partition_inclusive(np.arange(0, 4))
>>> P_x = P_x_base.create_cartesian_topology_partition([1, 4])
>>>
>>> x_local_shape = np.array([1, 40])
>>>
>>> criterion = DistributedMSELoss(P_x, reduction="mean")
>>>
>>> x = zero_volume_tensor()
>>> y = zero_volume_tensor()
>>> if P_x.active:
>>>     x = torch.rand(*x_local_shape)
>>>     y = torch.rand(*x_local_shape)
>>>
>>> loss = criterion(x, y)
>>>
>>> loss.backward()

API
===

.. currentmodule:: distdl.nn

.. autoclass:: DistributedLossBase
    :members:

.. autoclass:: DistributedL1Loss
    :members:

.. autoclass:: DistributedMSELoss
    :members:

.. autoclass:: DistributedPoissonNLLLoss
    :members:

.. autoclass:: DistributedBCELoss
    :members:

.. autoclass:: DistributedBCEWithLogitsLoss
    :members:

.. autoclass:: DistributedKLDivLoss
    :members:
