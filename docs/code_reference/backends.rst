.. role:: hidden
    :class: hidden-section

===============
distdl.backends
===============

.. toctree::
    :glob:
    :hidden:

    backends/*

.. contents::
    :local:
    :depth: 3


Overview
========


Data Movement Backend Interface
===============================

Partitions
----------

Partitions with No Topology
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Paritions with Cartesian Topology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functional Primitives
---------------------

Broadcast
~~~~~~~~~

The functional primitive for the Broadcast data movement operation does use
the original tensor partitions :math:`P_x` (input) and :math:`P_y` (output)
directly.  Instead, the calling class must create two new partitions to enable
actual data movement.

For the Broadcast operation, these are back-end specific implementations of
:ref:`Partitions with No Topology <Partitions with No Topology>` and the data
movement occurs *within* these partitions.

The partitions are created using the ``create_broadcast_partition_to()``
method of the ``Partition`` class for the selected back-end, which creates a
number of partitions equal to the product of the shape if :math:`P_x`.

Each of these partitions has, as its root, the worker that is the source of
the data to be copied in the broadcast and all other workers in the partition
are to receive those copies.  The root is specified in a back-end specific
manner, e.g., the first worker in the partition is the 0th rank in the
associated group of processors in the MPI back-end implementation.

Each worker in :math:`P_x` or :math:`P_y` has access to up to two of these
partitions.  If the worker is active in :math:`P_x` it will have a non-null
``P_send`` partition, which it is the root worker of.  If it is active in
:math:`P_y`, it will have a non-null ``P_recv`` partition.  Usually, the
worker will be a non-root worker in ``P_recv``, indicating that it is going to
receive a copy from the root.  However, it is possible that the worker *may be
the root of* ``P_recv``.  This occurs if ``P_send`` and ``P_recv`` are the
same and thus the worker is required to copy its own data to itself as part of
the broadcast.

We illustrate some examples of the construction of these partitions in the
following example.  Consider a broadcast of a tensor partitioned by a :math:`1
\times 3 \times 1` partition :math:`P_x` to :math:`2 \times 3 \times 2`
partition :math:`P_y`, as illustrated below, where workers are identified a
*global* lexicographic ordering.  For example, in an MPI-based environment,
this would mean the identifiers are the ranks in a group of size 12 that is a
parent group to both partitions.

.. note::
    Implementations should not impose anything on the ordering of the workers
    in :math:`P_x` and :math:`P_y`.  The sets of workers in these partitions
    may be disjoint, partially overlapping, or completely overlapping.


.. figure:: /_images/broadcast_example_A01.png
    :alt: Image of 1x3x1 to 2x3x1 broadcast with labeled workers.

    Blue, yellow, and red subtensors on :math:`P_x` are to be broadcast to
    :math:`P_y`.  Labels on subtensors are global lexicographic identifiers
    for the workers.


This broadcast creates 3 partitions as described above, illustrated below, which we
label :math:`P_0`, :math:`P_1`, and :math:`P_2` for convenience.


.. figure:: /_images/broadcast_example_A02.png
    :alt: Image of partitions induced by 1x3x1 to 2x3x1 broadcast, with labeled workers.

The following table gives the global lexicographic identifiers of each worker
in the partition, with the root worker listed first in bold.

.. list-table::
   :header-rows: 1
   :align: center

   * - Partition
     - Workers
   * - :math:`P_0`
     - **1**, 0, 6, 7
   * - :math:`P_1`
     - **2**, 3, 8, 9
   * - :math:`P_2`
     - **3**, 4, 5, 10, 11

The following table gives the partition label for the partition that each
worker associates with ``P_send`` and ``P_recv``.

.. list-table::
   :width: 95%
   :widths: 10 10 10 1 10 10 10  1 10 10 10
   :header-rows: 1
   :align: center

   * - Worker
     - ``P_send``
     - ``P_recv``
     -
     - Worker
     - ``P_send``
     - ``P_recv``
     -
     - Worker
     - ``P_send``
     - ``P_recv``
   * - 0
     - n/a
     - :math:`P_0`
     -
     - 4
     - n/a
     - :math:`P_2`
     -
     - 8
     - n/a
     - :math:`P_1`
   * - 1
     - :math:`P_0`
     - :math:`P_0`
     -
     - 5
     - n/a
     - :math:`P_2`
     -
     - 9
     - n/a
     - :math:`P_1`
   * - 2
     - :math:`P_1`
     - :math:`P_1`
     -
     - 6
     - n/a
     - :math:`P_0`
     -
     - 10
     - n/a
     - :math:`P_2`
   * - 3
     - :math:`P_2`
     - :math:`P_1`
     -
     - 7
     - n/a
     - :math:`P_0`
     -
     - 11
     - n/a
     - :math:`P_2`

This example illustrates the case where a worker needs to broadcast to itself
as root (workers 1 and 2), the case where a worker broadcasts from itself as
root and receives data in another partition (worker 3), and the case where a
worker simply receives copies of the data from the roots (workers 0 and 4-11).
It also illustrates that the root worker is the one that has the data, not the
lowest globally ranked worker (e.g., in :math:`P_0`).

After the forward application of the broadcast, all workers within each of the
partitions have as output a copy of the correct subtensor of the original
tensor.

.. figure:: /_images/broadcast_example_A03.png
    :alt: Image of 1x3x1 to 2x3x1 broadcast with labeled workers.

    Completed broadcast.

The adjoint phase works the same way, but in reverse.  However, as
demonstrated in the `motivating paper <https://arxiv.org/abs/2006.03108>`_,
the subtensors on :math:`P_y` are sum-reduced back to :math:`P_x`, rather than
copied.

Halo Exchange
~~~~~~~~~~~~~

Sum-Reduce
~~~~~~~~~~

Transpose
~~~~~~~~~

Backends
========

.. list-table::
   :widths: 35 65
   :width: 100%
   :header-rows: 0
   :align: left

   * - :ref:`code_reference/backends/mpi:MPI Backend`
     - The MPI data movement backend.
