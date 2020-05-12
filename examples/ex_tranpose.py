import numpy as np
import torch
from mpi4py import MPI

import distdl.nn.transpose as transpose
import distdl.utilities.slicing as slicing
from distdl.utilities.debug import print_sequential
from distdl.utilities.misc import Bunch

# Set up MPI cartesian communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

in_dims = (4, 1)
in_comm = comm.Create_cart(dims=in_dims)

in_rank = in_comm.Get_rank()
in_size = in_comm.Get_size()


out_dims = (2, 2)
out_comm = comm.Create_cart(dims=out_dims)

out_rank = out_comm.Get_rank()
out_size = out_comm.Get_size()


sizes = np.array([7, 5])
layer = transpose.DistributedTranspose(sizes, comm, in_comm, out_comm)

f = transpose.DistributedTransposeFunction

in_subsizes = slicing.compute_subsizes(in_comm.dims,
                                       in_comm.Get_coords(in_rank),
                                       sizes)
x = np.zeros(in_subsizes) + in_rank + 1

in_buffers, out_buffers = layer._allocate_buffers(x.dtype)

x = torch.from_numpy(x)
print_sequential(comm, f"x_{rank}: {x}")

ctx = Bunch()

y = f.forward(ctx, x, comm, sizes,
              layer.in_slices, in_buffers, in_comm,
              layer.out_slices, out_buffers, out_comm)
print_sequential(comm, f"y_{rank}: {y}")

out_subsizes = slicing.compute_subsizes(out_comm.dims,
                                        out_comm.Get_coords(out_rank),
                                        sizes)
gy = np.zeros(out_subsizes) + out_rank + 1
gy = torch.from_numpy(gy)
print_sequential(comm, f"gy_{rank}: {gy}")

gx = f.backward(ctx, gy)
print_sequential(comm, f"gx_{rank}: {gx}")
