# This example demonstrates the behavior of the halo-exchange primitive.
#
# It requires 4 workers to run.
#
# Run with, e.g.,
#     > mpirun -np 4 python ex_halo_exchange.py

import numpy as np
import torch
import torch.nn.functional as F
from mpi4py import MPI

from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.mixins.conv_mixin import ConvMixin
from distdl.nn.mixins.halo_mixin import HaloMixin
from distdl.utilities.slicing import compute_subshape


# We need a layer to induce the halo size.  To make this convenient, we mock
# a layer that has the right mixins to do the trick.
class MockConvLayer(HaloMixin, ConvMixin):
    pass


# Setup a standard feature-space convolution kernel.
mockup_conv_layer = MockConvLayer()
kernel_size = [1, 1, 3, 3]
stride = [1, 1, 1, 1]
padding = [0, 0, 0, 0]
dilation = [1, 1, 1, 1]

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# Create the input partition (using the first 4 workers)
in_shape = (1, 1, 2, 2)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# Pick a global shape that is big enough to require a halo.
x_global_shape = np.array([1, 1, 5, 4])

# Compute the necessary information about the exchange buffer sizes
exchange_info = mockup_conv_layer._compute_exchange_info(x_global_shape,
                                                         kernel_size,
                                                         stride,
                                                         padding,
                                                         dilation,
                                                         P_x.active,
                                                         P_x.shape,
                                                         P_x.index)
halo_shape = exchange_info[0]
recv_buffer_shape = exchange_info[1]
send_buffer_shape = exchange_info[2]

x_local_shape = compute_subshape(P_x.shape, P_x.index, x_global_shape)

# Input tensor will be (on a 1 x 1 x 2 x 2 partition):
# [ [ [ [ 1, 1, 0],      | [ [ [ [ 0, 2, 2],
#       [ 1, 1, 0],      |       [ 0, 2, 2],
#       [ 1, 1, 0],      |       [ 0, 2, 2],
#       [ 0, 0, 0] ] ] ] |       [ 0, 0, 0] ] ] ]
#       -----------------------------------
# [ [ [ [ 0, 0, 0],      | [ [ [ [ 0, 0, 0],
#       [ 3, 3, 0],      |       [ 0, 4, 4],
#       [ 3, 3, 0] ] ] ] |       [ 0, 4, 4] ] ] ]
x = np.zeros(x_local_shape) + (P_x.rank + 1)
x = torch.from_numpy(x)
x.requires_grad = True
# x has to be padded to make space for the halo.  Torch's pad function takes
# its arguments in very specific order and for pooling and convolutions,
# sometimes we need to trick the layers to do the computation the way we
# want.  So sometimes the padding only partially comes from the halo
# exchange.  Here it is from the halo exchange only.  The next operation
# gets the padding in the format that the torch pad function requires.
torch_padding = tuple(np.array(list(reversed(halo_shape)), dtype=int).flatten())

# We pad with "constant" mode here because it matches our internal behavior.
x = F.pad(x, pad=torch_padding, mode="constant", value=0)

# Not a leaf tensor (can't be because halo exchange is in-place) so
# we need to retain its gradient to see the adjoint effect later
x.retain_grad()

print(f"rank {P_world.rank}; index {P_x.index}; value {x.to(int)}")

# Define the exchange itself.  We let the operator define its own buffers,
# though we could give it a buffer manager.
halo_layer = HaloExchange(P_x, halo_shape, recv_buffer_shape, send_buffer_shape)

# Output tensor will be (on a 1 x 1 x 2 x 2 partition):
# [ [ [ [ 1, 1, 2],      | [ [ [ [ 1, 2, 2],
#       [ 1, 1, 2],      |       [ 1, 2, 2],
#       [ 1, 1, 2],      |       [ 1, 2, 2],
#       [ 3, 3, 4] ] ] ] |       [ 3, 4, 4] ] ] ]
#       -----------------------------------
# [ [ [ [ 1, 1, 2],      | [ [ [ [ 1, 2, 2],
#       [ 3, 3, 4],      |       [ 3, 4, 4],
#       [ 3, 3, 4] ] ] ] |       [ 3, 4, 4] ] ] ]
y = halo_layer(x)

print(f"rank {P_world.rank}; index {P_x.index}; value {y.to(int)}")


# Setup the adjoint input tensor.  Here we setup
#
# Adjoint input tensor will be (on a 1 x 1 x 2 x 2 partition):
# [ [ [ [ 1, 1, 1],      | [ [ [ [ 2, 2, 2],
#       [ 1, 1, 1],      |       [ 2, 2, 2],
#       [ 1, 1, 1],      |       [ 2, 2, 2],
#       [ 1, 1, 1] ] ] ] |       [ 2, 2, 2] ] ] ]
#       -----------------------------------
# [ [ [ [ 3, 3, 3],      | [ [ [ [ 4, 4, 4],
#       [ 3, 3, 3],      |       [ 4, 4, 4],
#       [ 3, 3, 3] ] ] ] |       [ 4, 4, 4] ] ] ]
dy = torch.zeros(y.shape) + (P_x.rank + 1)

print(f"rank {P_world.rank}; index {P_x.index}; value {dy}")

# Apply the adjoint of the layer.
#
# Adjoint output tensor will be (on a 1 x 1 x 2 x 2 partition):
# [ [ [ [ 1,  3, 0],      | [ [ [ [ 0,  3, 2],
#       [ 1,  3, 0],      |       [ 0,  3, 2],
#       [ 4, 10, 0],      |       [ 0, 10, 6],
#       [ 0,  0, 0] ] ] ] |       [ 0,  0, 0] ] ] ]
#       -----------------------------------
# [ [ [ [ 0,  0, 0],      | [ [ [ [ 0,  0, 0],
#       [ 4, 10, 0],      |       [ 0, 10, 6],
#       [ 3,  7, 0] ] ] ] |       [ 0,  7, 4] ] ] ]
y.backward(dy)
dx = x.grad
print(f"rank {P_world.rank}; index {P_x.index}; value {dx}")
