import numpy as np
from mpi4py import MPI

from distdl.utilities.debug import print_sequential


class MPIPartition:

    null_comm = MPI.COMM_NULL
    null_group = MPI.GROUP_NULL
    null_rank = MPI.PROC_NULL

    def __init__(self, comm, parent_partition=None):

        self.comm = comm
        self.active = (comm != MPI.COMM_NULL)

        self.group = comm.Get_group() if self.active else MPI.GROUP_NULL
        self.rank = comm.Get_rank() if self.active else MPI.PROC_NULL
        self.size = comm.Get_size() if self.active else 0

        self.parent_partition = parent_partition

    def create_subpartition(self, ranks):

        if self.active:
            g = self.group.Incl(ranks)
            c = self.comm.Create(g)
        else:
            c = MPI.COMM_NULL

        return MPIPartition(c, self)

    def create_cartesian_subpartition(self, dims, **options):

        if self.active:
            if np.prod(dims) != self.size:
                raise ValueError("Base wrapper must be of size prod(dims).")
            c = self.comm.Create_cart(dims, **options)
        else:
            c = MPI.COMM_NULL

        return MPICartesianPartition(c, self, dims)

    def ranks(self):

        for r in range(self.size):
            yield r

    def info(self, tag=None):

        if not self.active:
            return

        tag_str = ""
        if tag is not None:
            tag_str = f"{tag}:\t"

        print_sequential(self.comm,
                         f"{tag_str}{self.rank}/{self.size}")

    def common_ancestor(self, other):

        my_lineage = self.lineage()
        other_lineage = other.lineage()

        for w1 in my_lineage:
            for w2 in other_lineage:
                if w1.comm is w2.comm and w1.comm is not MPI.COMM_NULL:
                    return w1
        return None

    def lineage(self):
        if self.parent_partition is None:
            return [self]
        else:
            return [self] + self.parent_partition.lineage()


class MPICartesianPartition(MPIPartition):

    def __init__(self, comm, parent_partition, dims):

        super(MPICartesianPartition, self).__init__(comm, parent_partition)

        self.dims = dims

    def cartesian_coordinates(self, rank):

        if not self.active:
            raise Exception()

        return self.comm.Get_coords(rank)
