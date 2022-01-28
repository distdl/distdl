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

All Sum-Reduce
--------------

.. automodule:: distdl.backends.mpi.functional.all_sum_reduce

.. autoclass:: distdl.backends.mpi.functional.AllSumReduceFunction
    :members:
    :undoc-members:

Broadcast
---------

.. automodule:: distdl.backends.mpi.functional.broadcast

.. autoclass:: distdl.backends.mpi.functional.BroadcastFunction
    :members:
    :undoc-members:

Halo Exchange
-------------

.. automodule:: distdl.backends.mpi.functional.halo_exchange

.. autoclass:: distdl.backends.mpi.functional.HaloExchangeFunction
    :members:
    :undoc-members:

Sum-Reduce
----------

.. automodule:: distdl.backends.mpi.functional.sum_reduce

.. autoclass:: distdl.backends.mpi.functional.SumReduceFunction
    :members:
    :undoc-members:

Repartition
-----------

.. automodule:: distdl.backends.mpi.functional.repartition

.. autoclass:: distdl.backends.mpi.functional.RepartitionFunction
    :members:
    :undoc-members:

Misc
====

.. automodule:: distdl.backends.mpi.compare
    :members:
    :undoc-members:
