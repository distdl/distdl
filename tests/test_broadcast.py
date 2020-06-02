def test_potentially_deadlocked_send_recv_pairs():

    import numpy as np
    from mpi4py import MPI

    from distdl.backends.mpi.partition import MPIPartition
    from distdl.nn.broadcast import Broadcast

    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    P_8 = P_world.create_partition_inclusive(np.arange(4, 12))
    P_16 = P_world.create_partition_inclusive(np.arange(0, 16))

    P_x = P_8.create_cartesian_topology_partition([1, 2, 2, 2])
    P_w = P_16.create_cartesian_topology_partition([2, 2, 2, 2])

    layer = Broadcast(P_x, P_w)  # noqa F841

    P_world.comm.Barrier()
