import numpy as np
from mpi4py import MPI

from distdl.backends.mpi.compare import check_identical_comm
from distdl.backends.mpi.compare import check_identical_group
from distdl.backends.mpi.compare import check_null_comm
from distdl.backends.mpi.compare import check_null_group
from distdl.backends.mpi.compare import check_null_rank
from distdl.utilities.debug import print_sequential
from distdl.utilities.index_tricks import cartesian_index


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

    def create_broadcast_partition_to(self, P_dest):

        P_src = self

        P_send = MPIPartition(MPI.COMM_NULL)
        P_recv = MPIPartition(MPI.COMM_NULL)

        P_union = MPIPartition(MPI.COMM_NULL)
        if P_src.active or P_dest.active:
            P_union = P_src.create_partition_union(P_dest)

        # If we are not active in one of the two partitions, return null
        # partitions
        if not P_union.active:
            return P_send, P_recv

        # Find the rank in P_union with rank 0 of P_src
        rank_map_data = np.array([-1], dtype=np.int)
        if P_src.active:
            rank_map_data[0] = P_src.rank
        rank_map = -1*np.ones(P_union.size, dtype=np.int)
        P_union.comm.Allgather(rank_map_data, rank_map)
        src_root = np.where(rank_map == 0)[0][0]

        # Find the rank in P_union with rank 0 of P_dest
        rank_map_data = np.array([-1], dtype=np.int)
        if P_dest.active:
            rank_map_data[0] = P_dest.rank
        rank_map = -1*np.ones(P_union.size, dtype=np.int)
        P_union.comm.Allgather(rank_map_data, rank_map)
        dest_root = np.where(rank_map == 0)[0][0]

        # Share the src cartesian dimension with everyone
        src_dim = np.zeros(1, dtype=np.int)
        if P_src.active and P_src.rank == 0:
            src_dim[0] = P_src.dim
        P_union.comm.Bcast(src_dim, root=src_root)

        # Share the dest cartesian dimension with everyone
        dest_dim = np.zeros(1, dtype=np.int)
        if P_dest.active and P_dest.rank == 0:
            dest_dim[0] = P_dest.dim
        P_union.comm.Bcast(dest_dim, root=dest_root)

        # The source must be smaller (or equal) in size to the destination.
        if src_dim > dest_dim:
            raise Exception("No broadcast: Source partition larger than "
                            "destination partition.")

        # Share the src partition dimensions with everyone.  We will compare
        # this with the destination dimensions, so we pad it to the left with
        # ones to make a valid comparison.
        src_dims = np.ones(dest_dim, dtype=np.int)
        if P_src.active and P_src.rank == 0:
            src_dims[-src_dim[0]:] = P_src.dims
        P_union.comm.Bcast(src_dims, root=src_root)

        # Share the dest partition dimensions with everyone
        dest_dims = np.zeros(dest_dim, dtype=np.int)
        if P_dest.active and P_dest.rank == 0:
            dest_dims = P_dest.dims
        P_union.comm.Bcast(dest_dims, root=dest_root)

        # Find any location that the dimensions differ and where the source
        # dimension is not 1 where they differ.  If there are any such
        # dimensions, we cannot perform a valid broadcast.
        no_match_loc = np.where((src_dims != dest_dims) & (src_dims != 1))[0]

        if len(no_match_loc) > 0:
            raise Exception("No broadcast: Dimensions don't match or "
                            "source is not 1 where there is a mismatch.")

        # We will use the matching dimensions to compute the broadcast indices
        match_loc = np.where((src_dims == dest_dims))[0]

        # Compute the Cartesian index of the source rank, in the matching
        # dimensions only.  This index will be constant in the dimensions of
        # the destination that we are broadcasting along.
        src_index = -1
        if P_src.active:
            coords_src = np.ones_like(src_dims)
            coords_src[-src_dim[0]:] = P_src.cartesian_coordinates(P_src.rank)
            src_index = cartesian_index(src_dims[match_loc],
                                        coords_src[match_loc])

        # Compute the Cartesian index of the destination rank, in the matching
        # dimensions only.  This index will match the index in the source we
        # receive the broadcast from.
        dest_index = -1
        if P_dest.active:
            coords_dest = P_dest.cartesian_coordinates(P_dest.rank)
            dest_index = cartesian_index(dest_dims[match_loc],
                                         coords_dest[match_loc])

        # Share the two indices with every worker in the union.  The first
        # column of data contains the source "index" and the second contains
        # the destination "index".
        union_indices = -1*np.ones(2*P_union.size, dtype=np.int)
        local_indices = np.array([src_index, dest_index], dtype=np.int)
        P_union.comm.Allgather(local_indices, union_indices)
        union_indices.shape = (-1, 2)

        # Build partitions to communicate single broadcasts across subsets
        # of the union partition.

        # For the sending partition, the destination ranks are the ones that
        # have matching Cartesian indices in the matching dimensions.
        send_root = MPI.PROC_NULL
        if P_src.active:
            # The ranks in the union that I will send data to
            send_dests = np.where(union_indices[:, 1] == src_index)[0]
            # My rank in the union
            send_root = P_union.rank

        # For the receiving partition, if I am a destination, the other
        # destinations are the ones with the same Cartesian index as me.
        recv_root = MPI.PROC_NULL
        if P_dest.active:
            # The ranks in the union that will also receive data with me
            recv_dests = np.where(union_indices[:, 1] == dest_index)[0]
            # The rank in the union I will receive data from
            recv_root = np.where(union_indices[:, 0] == dest_index)[0][0]

        # Create the MPI group for the broadcast, if any, for which I am root.
        send_ranks = []
        group_send = MPI.GROUP_NULL
        if send_root != MPI.PROC_NULL:
            # Ensure that ranks are not repeated in the union
            send_ranks = [rank for rank in send_dests if rank != send_root]
            # Ensure that the root rank is first, so it will be rank 0 in the
            # new communicator
            send_ranks = [send_root] + send_ranks
            send_ranks = np.array(send_ranks)
            group_send = P_union.group.Incl(send_ranks)

        # Create the MPI group for the broadcast, if any, for which I receive
        # data.
        recv_ranks = []
        group_recv = MPI.GROUP_NULL
        if recv_root != MPI.PROC_NULL:
            # Ensure that ranks are not repeated in the union
            recv_ranks = [rank for rank in recv_dests if rank != recv_root]
            # Ensure that the root rank is first, so it will be rank 0 in the
            # new communicator
            recv_ranks = [recv_root] + recv_ranks
            recv_ranks = np.array(recv_ranks)
            group_recv = P_union.group.Incl(recv_ranks)

        # We will only do certain work if certain groups were created.
        has_send_group = not check_null_group(group_send)
        has_recv_group = not check_null_group(group_recv)
        same_send_recv_group = check_identical_group(group_send, group_recv)

        if has_send_group:
            comm_send = P_union.comm.Create_group(group_send)
            P_send = MPIPartition(comm_send, group_send,
                                  root=P_union.root)
        if same_send_recv_group:
            P_recv = P_send
        elif has_recv_group:
            comm_recv = P_union.comm.Create_group(group_recv)
            P_recv = MPIPartition(comm_recv, group_recv,
                                  root=P_union.root)

        return P_send, P_recv

    def create_reduction_partition_to(self, P_dest):

        P_src = self

        P_same = MPIPartition(MPI.COMM_NULL)
        P_send = MPIPartition(MPI.COMM_NULL)
        P_recv = MPIPartition(MPI.COMM_NULL)

        P_union = MPIPartition(MPI.COMM_NULL)
        if P_src.active or P_dest.active:
            P_union = P_src.create_partition_union(P_dest)

        # If we are not active in one of the two partitions, return null
        # partitions
        if not P_union.active:
            return P_same, P_send, P_recv

        # Find the rank in P_union with rank 0 of P_src
        rank_map_data = np.array([-1], dtype=np.int)
        if P_src.active:
            rank_map_data[0] = P_src.rank
        rank_map = -1*np.ones(P_union.size, dtype=np.int)
        P_union.comm.Allgather(rank_map_data, rank_map)
        src_root = np.where(rank_map == 0)[0][0]

        # Find the rank in P_union with rank 0 of P_dest
        rank_map_data = np.array([-1], dtype=np.int)
        if P_dest.active:
            rank_map_data[0] = P_dest.rank
        rank_map = -1*np.ones(P_union.size, dtype=np.int)
        P_union.comm.Allgather(rank_map_data, rank_map)
        dest_root = np.where(rank_map == 0)[0][0]

        # Share the src cartesian dimension with everyone
        src_dim = np.zeros(1, dtype=np.int)
        if P_src.active and P_src.rank == 0:
            src_dim[0] = P_src.dim
        P_union.comm.Bcast(src_dim, root=src_root)

        # Share the dest cartesian dimension with everyone
        dest_dim = np.zeros(1, dtype=np.int)
        if P_dest.active and P_dest.rank == 0:
            dest_dim[0] = P_dest.dim
        P_union.comm.Bcast(dest_dim, root=dest_root)

        # The source must be smaller (or equal) in size to the destination.
        if dest_dim > src_dim:
            raise Exception("No reduction: Source partition smaller than "
                            "destination partition.")

        # Share the src partition dimensions with everyone
        src_dims = np.zeros(src_dim, dtype=np.int)
        if P_src.active and P_src.rank == 0:
            src_dims = P_src.dims
        P_union.comm.Bcast(src_dims, root=src_root)

        # Share the dest partition dimensions with everyone.  We will compare
        # this with the source dimensions, so we pad it to the left with
        # ones to make a valid comparison.
        dest_dims = np.ones(src_dim, dtype=np.int)
        if P_dest.active and P_dest.rank == 0:
            dest_dims[-dest_dim[0]:] = P_dest.dims
        P_union.comm.Bcast(dest_dims, root=dest_root)

        # Find any location that the dimensions differ and where the dest
        # dimension is not 1 where they differ.  If there are any such
        # dimensions, we cannot perform a valid reduction.
        no_match_loc = np.where((src_dims != dest_dims) & (dest_dims != 1))[0]

        if len(no_match_loc) > 0:
            raise Exception("No broadcast: Dimensions don't match or "
                            "source is not 1 where there is a mismatch.")

        # We will use the matching dimensions to compute the broadcast indices
        match_loc = np.where((src_dims == dest_dims))[0]

        # Compute the Cartesian index of the source rank, in the matching
        # dimensions only.  This index will be constant in the dimensions of
        # the destination that we are reducing along.
        src_index = -1
        if P_src.active:
            coords_src = P_src.cartesian_coordinates(P_src.rank)
            src_index = cartesian_index(src_dims[match_loc],
                                        coords_src[match_loc])

        # Compute the Cartesian index of the destination rank, in the matching
        # dimensions only.  This index will match the index in the source we
        # receive the broadcast from.
        dest_index = -1
        if P_dest.active:
            coords_dest = np.ones_like(dest_dims)
            coords_dest[-dest_dim[0]:] = P_dest.cartesian_coordinates(P_dest.rank)
            dest_index = cartesian_index(dest_dims[match_loc],
                                         coords_dest[match_loc])

        # Share the two indices with every worker in the union.  The first
        # column of data contains the source "index" and the second contains
        # the destination "index".
        union_indices = -1*np.ones(2*P_union.size, dtype=np.int)
        local_indices = np.array([src_index, dest_index], dtype=np.int)
        P_union.comm.Allgather(local_indices, union_indices)
        union_indices.shape = (-1, 2)

        # Build partitions to communicate single reductions across subsets
        # of the union partition.

        # For the sending partition, the destination ranks are the ones that
        # have matching Cartesian indices in the matching dimensions.
        send_root = MPI.PROC_NULL
        send_sources = None
        if P_src.active:
            # The ranks in the union that sending data to the same place as me
            send_sources = np.where(union_indices[:, 0] == src_index)[0]
            # The rank in the union I send data to
            send_root = np.where(union_indices[:, 1] == src_index)[0][0]

        # For the receiving partition, if I am a destination, the other
        # destinations are the ones with the same Cartesian index as me.
        recv_root = MPI.PROC_NULL
        recv_sources = None
        if P_dest.active:
            # The ranks in the union that send data to me
            recv_sources = np.where(union_indices[:, 0] == dest_index)[0]
            # My rank in the union
            recv_root = P_union.rank

        # Create the MPI group for the reduction, if any, for which I send
        # data.
        send_ranks = []
        group_send = MPI.GROUP_NULL
        if send_root != MPI.PROC_NULL:
            # Ensure that ranks are not repeated in the union
            send_ranks = [rank for rank in send_sources if rank != send_root]
            # Ensure that the root rank is first, so it will be rank 0 in the
            # new communicator
            send_ranks = [send_root] + send_ranks
            send_ranks = np.array(send_ranks)
            group_send = P_union.group.Incl(send_ranks)

        # Create the MPI group for the reduction, if any, for which I am root.
        recv_ranks = []
        group_recv = MPI.GROUP_NULL
        if recv_root != MPI.PROC_NULL:
            # Ensure that ranks are not repeated in the union
            recv_ranks = [rank for rank in recv_sources if rank != recv_root]
            # Ensure that the root rank is first, so it will be rank 0 in the
            # new communicator
            recv_ranks = [recv_root] + recv_ranks
            recv_ranks = np.array(recv_ranks)
            group_recv = P_union.group.Incl(recv_ranks)

        # We will only do certain work if certain groups were created.
        has_send_group = not check_null_group(group_send)
        has_recv_group = not check_null_group(group_recv)
        same_send_recv_group = check_identical_group(group_send, group_recv)

        # If the send and receive group are the same, we will deadlock making
        # two communicators (via MPIPartition).  So we only create one.
        if same_send_recv_group:
            comm_common = P_union.comm.Create_group(group_send)
            group_common = comm_common.Get_group()
            P_same = MPIPartition(comm_common, group_common,
                                  root=P_union.root)
        # Otherwise, we create separate partitions for sending and receiving
        else:
            if has_send_group:
                comm_send = P_union.comm.Create_group(group_send)
                P_send = MPIPartition(comm_send, group_send,
                                      root=P_union.root)
            if has_recv_group:
                comm_recv = P_union.comm.Create_group(group_recv)
                P_recv = MPIPartition(comm_recv, group_recv,
                                      root=P_union.root)

        return P_same, P_send, P_recv


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
