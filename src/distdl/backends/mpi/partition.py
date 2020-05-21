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

    def __eq__(self, other):
        return MPI.Comm.Compare(self.comm, other.comm) == MPI.IDENT

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

    def map_from_ancestor(self, ancestor):

        if ancestor not in self.lineage():
            raise Exception("'ancestor' is not an ancesor of self.")

        ancestor_to_self = np.zeros(ancestor.size, dtype=np.int)
        v = -1 if self.rank is MPI.PROC_NULL else self.rank
        ancestor.comm.Allgather(np.array([v]), ancestor_to_self)

        return ancestor_to_self


class MPICartesianPartition(MPIPartition):

    def __init__(self, comm, parent_partition, dims):

        super(MPICartesianPartition, self).__init__(comm, parent_partition)

        self.dims = dims
        self.dim = len(dims)

    def cartesian_coordinates(self, rank):

        if not self.active:
            raise Exception()

        return self.comm.Get_coords(rank)

    def neighbor_ranks(self, rank):

        if not self.active:
            raise Exception()

        coords = self.cartesian_coordinates(rank)

        # Resulting list
        neighbor_ranks = []

        # Loop over the dimensions and add the ranks at the neighboring coords to the list
        for i in range(self.dim):
            lcoords = [x-1 if j == i else x for j, x in enumerate(coords)]
            rcoords = [x+1 if j == i else x for j, x in enumerate(coords)]
            lrank = MPI.PROC_NULL if -1 == lcoords[i] else self.comm.Get_cart_rank(lcoords)
            rrank = MPI.PROC_NULL if self.dims[i] == rcoords[i] else self.comm.Get_cart_rank(rcoords)
            neighbor_ranks.append((lrank, rrank))

        return neighbor_ranks

    def create_subpartition(self, remain_dims):

        c = self.Sub(remain_dims)
        return MPICartesianPartition(c, self, c.dims)
