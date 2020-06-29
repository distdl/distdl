===========
MPI Backend
===========

.. contents::
    :local:
    :depth: 3

Overview
========

Tensor Partitions
=================

.. currentmodule:: distdl.backends.mpi

.. autoclass:: Partition

.. autoclass:: CartesianPartition

.. automodule:: distdl.backends.mpi.partition
    :members:
    :undoc-members:

.. automodule:: distdl.backends.mpi.tensor_comm
    :members:
    :undoc-members:

Primitive Functionals
=====================

Broadcast
---------

.. autoclass:: distdl.backends.mpi.autograd.BroadcastFunction
    :members:
    :undoc-members:

Halo Exchange
-------------

.. automodule:: distdl.backends.mpi.autograd.halo_exchange
    :members:
    :undoc-members:

Sum-Reduce
----------

.. automodule:: distdl.backends.mpi.autograd.sum_reduce
    :members:
    :undoc-members:

Transpose
---------

.. automodule:: distdl.backends.mpi.autograd.transpose
    :members:
    :undoc-members:

Misc
====

.. automodule:: distdl.backends.mpi.compare
    :members:
    :undoc-members:
