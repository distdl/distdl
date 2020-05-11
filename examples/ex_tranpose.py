import numpy as np
from mpi4py import MPI

import distdl.nn.transpose as transpose

# Set up MPI cartesian communicator
comm = MPI.COMM_WORLD
in_dims = (4, 1)
in_comm = comm.Create_cart(dims=in_dims)

in_rank = in_comm.Get_rank()
in_size = in_comm.Get_size()


out_dims = (2, 2)
out_comm = comm.Create_cart(dims=out_dims)

out_rank = out_comm.Get_rank()
out_size = out_comm.Get_size()


sizes = np.array([4, 4])
result = transpose.DistributedTranspose(sizes, comm, in_comm, out_comm)
