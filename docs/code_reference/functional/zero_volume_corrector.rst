=====================
Zero Volume Corrector
=====================

.. contents::
    :local:
    :depth: 2


Overview
========

Some aspects of PyTorch cannot handle zero-volume tensors.  For example, loss
functions cannot back-propagate through non-scalar outputs.  In these cases,
DistDL layers that return zero-volume tensors are incompatible. The
zero-volume corrector function provides a work-around.

Implementation
==============

Any non-zero-volume tensor input is returned directly.  Any zero-volume input
is replaced with a dummy, but valid, tensor.  In the corresponding adjoint
function, the gradient output is returned for those workers that had
non-zero-volume inputs and a zero-volume gradient input is returned for
workers whose forward inputs were zero-volume.  No information about the
tensor's partition is required.

API
===

.. currentmodule:: distdl.functional

.. autoclass:: ZeroVolumeCorrectorFunction
    :members:
