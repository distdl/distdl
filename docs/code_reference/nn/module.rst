=======================
Base Distributed Module
=======================

.. contents::
    :local:
    :depth: 2

Overview
========

The :any:`Module` container is an extension of the Torch :ref:`torch.nn.Module
<torch:torch.nn.Module>` container which defines the interface that is
required to allow a DistDL distributed layer to perform any necessary setup or
teardown operations.

.. note::

   This container inherits from PyTorch and should be used as the base-class
   for any DistDL distributed layer.

Motivation
==========

DistDL aims to preserve PyTorch-like interfaces, so that information about the
input tensor, other than the partition functions, is not required at the
instantiation of the layer.  Consequently, some layer variables, such as
intermediate communication buffers, for example in the
:any:`distdl.nn.HaloExchange` layer or the
:any:`distdl.nn.Repartition` layer, can only be determined when the
layer is evaluated.

The interfaces defined here allow those properties to be setup (and torn down)
safely when the layer is evaluated.

Implementation
==============

However, there is significant cost to the setup phase.  To avoid this cost,
the setup phase is only called when there is a change to the structure of the
*global* input tensor.  We currently define the structure to be the shape and
``requires_grad`` status.  In the future, the ``dtype`` will also be part of
this determination.

.. warning::
   Each worker will only have knowledge of their local input tensors.  It is
   not possible to determine if the global tensor has changed without a global
   communication. Practically, this means that the first dimension of the
   tensor, usually the batch dimension, which should be the same across all
   workers in a partition, is the primary indicator of change.

   It is possible that the feature shape can be changed globally without
   changing the local feature shape, so care must be taken.

.. warning::
   This is largely designed to allow the batch to change, however, for
   performance purposes, frequent changes to the batch size and other tensor
   structure should be avoided.

The check if the input tensor has changed is defined per-layer, allowing for
implementation specific behavior.

The :any:`Module._distdl_forward_pre_hook` function is registered as a Torch
pre-forward hook.  This hook checks if the layer needs to be setup, either
from scratch or as a reset, and then calls the setup function.

The :any:`Module` container does not implement any of the following operations,
but the class does define the required interfaces.

API
===

.. currentmodule:: distdl.nn.module

.. autoclass:: Module
    :members:
    :private-members:
