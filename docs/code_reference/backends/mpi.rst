===========
MPI Backend
===========

Overview
========

.. All of these should be written directly here, not in the code doc string.
.. This way, we can use automodule without extraneous crap.
.. .. automodule:: distdl.backends.mpi

Tensor Partitions
=================

.. .. automodule:: distdl.backends.mpi.partition

Primitive Functionals
=====================

Broadcast
---------

.. .. automodule:: distdl.backends.mpi.autograd.broadcast

Halo Exchange
-------------

.. .. automodule:: distdl.backends.mpi.autograd.halo_exchange

Sum-Reduce
----------

.. .. automodule:: distdl.backends.mpi.autograd.sum_reduce

Transpose
---------

.. .. automodule:: distdl.backends.mpi.autograd.transpose


API
===

.. currentmodule:: distdl.backends.mpi

.. autoclass:: Partition

.. autoclass:: CartesianPartition

.. automodule:: distdl.backends.mpi.partition
    :members:
    :undoc-members:

.. automodule:: distdl.backends.mpi.compare
    :members:
    :undoc-members:

.. automodule:: distdl.backends.mpi.tensor_comm
    :members:
    :undoc-members:

.. automodule:: distdl.backends.mpi.autograd.broadcast
    :members:
    :undoc-members:

.. automodule:: distdl.backends.mpi.autograd.halo_exchange
    :members:
    :undoc-members:

.. automodule:: distdl.backends.mpi.autograd.sum_reduce
    :members:
    :undoc-members:

.. automodule:: distdl.backends.mpi.autograd.transpose
    :members:
    :undoc-members:
