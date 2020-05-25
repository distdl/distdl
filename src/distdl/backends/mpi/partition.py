import numpy as np
from mpi4py import MPI

from distdl.backends.mpi.compare import check_identical_comm
from distdl.backends.mpi.compare import check_identical_group
from distdl.backends.mpi.compare import check_null_comm
from distdl.backends.mpi.compare import check_null_group
from distdl.backends.mpi.compare import check_null_rank
from distdl.utilities.debug import print_sequential


class MPIPartition:

    def __init__(self, comm, group=MPI.GROUP_NULL, root=None):

        self.comm = comm

        # root tracks a root communicator: any subpartition from this one
        # will have the same root as this one.
        if root is None:
            self.root = comm
        else:
            self.root = root

        if self.comm != MPI.COMM_NULL:
            self.active = True
            if group == MPI.GROUP_NULL:
                self.group = comm.Get_group()
            else:
                self.group = group
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.active = False
            self.group = group
            self.rank = MPI.PROC_NULL
            self.size = -1

    def __eq__(self, other):

        # MPI_COMM_NULL is not a valid argument to MPI_Comm_compare, per the
        # MPI spec.  Because reasons.
        # We will require two partitions to have MPI_IDENT communicators to
        # consider them to be equal.
        if (check_null_comm(self.comm) or
            check_null_comm(other.comm) or
            check_null_group(self.group) or
            check_null_group(other.group) or
            check_null_rank(self.rank) or
            check_null_rank(other.rank)): # noqa E129
            return False

        return (check_identical_comm(self.comm, other.comm) and
                check_identical_group(self.group, other.group) and
                self.rank == other.rank)

    def print_sequential(self, val):

        if self.active:
            print_sequential(self.comm, val)

    def create_partition_inclusive(self, ranks):

        ranks = np.asarray(ranks)
        group = self.group.Incl(ranks)

        comm = self.comm.Create_group(group)

        return MPIPartition(comm, group, root=self.root)

    def create_partition_union(self, other):

        # Cannot make a union if the two partitions do not share a root
        if not check_identical_comm(self.root, other.root):
            raise Exception()

        group = MPI.Group.Union(self.group, other.group)

        comm = self.root.Create_group(group)

        return MPIPartition(comm, group, root=self.root)

    def create_cartesian_topology_partition(self, dims, **options):

        dims = np.asarray(dims)
        if self.active:
            comm = self.comm.Create_cart(dims, **options)
            group = comm.Get_group()

            if not check_identical_group(self.group, group):
                raise Exception()

            # group = self.group
            return MPICartesianPartition(comm, group, self.root, dims)

        else:
            comm = MPI.COMM_NULL
            return MPIPartition(comm, self.group, root=self.root)


class MPICartesianPartition(MPIPartition):

    def __init__(self, comm, group, root, dims):

        super(MPICartesianPartition, self).__init__(comm, group, root)

        self.dims = np.asarray(dims).astype(np.int)
        self.dim = len(self.dims)

        self.coords = None
        if self.active:
            self.coords = self.cartesian_coordinates(self.rank)

    def create_cartesian_subtopology_partition(self, remain_dims):

        # remain_dims = np.asarray(remain_dims)
        if self.active:
            comm = self.comm.Sub(remain_dims)
            group = comm.Get_group()

            return MPICartesianPartition(comm, group,
                                         self.root,
                                         self.dims[remain_dims == True]) # noqa E712

        else:
            comm = MPI.COMM_NULL
            return MPIPartition(comm, root=self.root)

    def cartesian_coordinates(self, rank):

        if not self.active:
            raise Exception()

        return np.asarray(self.comm.Get_coords(rank))

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
