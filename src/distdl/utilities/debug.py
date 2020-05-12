import sys


def print_sequential(comm, val):
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(f'{val}')
        for i in range(1, size):
            val = comm.recv(source=i, tag=0)
            print(f'{val}')
        sys.stdout.flush()
    else:
        comm.send(val, dest=0, tag=0)
