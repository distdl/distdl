from mpi4py import MPI


def get_neighbor_ranks(cart_comm):

    # Get comm info
    dim = cart_comm.dim
    dims = cart_comm.dims
    rank = cart_comm.Get_rank()
    coords = cart_comm.Get_coords(rank)

    # Resulting list
    neighbor_ranks = []

    # Loop over the dimensions and add the ranks at the neighboring coords to the list
    for i in range(dim):
        lcoords = [x-1 if j == i else x for j, x in enumerate(coords)]
        rcoords = [x+1 if j == i else x for j, x in enumerate(coords)]
        lrank = MPI.PROC_NULL if -1 == lcoords[i] else cart_comm.Get_cart_rank(lcoords)
        rrank = MPI.PROC_NULL if dims[i] == rcoords[i] else cart_comm.Get_cart_rank(rcoords)
        neighbor_ranks.append((lrank, rrank))

    return neighbor_ranks
