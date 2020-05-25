from mpi4py import MPI


def check_identical_comm(c1, c2):
    return MPI.Comm.Compare(c1, c2) == MPI.IDENT


def check_identical_group(g1, g2):
    return MPI.Group.Compare(g1, g2) == MPI.IDENT


def check_null_comm(c):
    return c == MPI.COMM_NULL


def check_null_group(g):
    return g == MPI.GROUP_NULL


def check_null_rank(r):
    return r == MPI.PROC_NULL
