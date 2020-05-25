from mpi4py import MPI


def check_identical_comm(c1, c2):
    if check_null_comm(c1) or check_null_comm(c2):
        return False
    return MPI.Comm.Compare(c1, c2) == MPI.IDENT


def check_identical_group(g1, g2):
    if check_null_group(g1) or check_null_group(g2):
        return False
    return MPI.Group.Compare(g1, g2) == MPI.IDENT


def check_null_comm(c):
    return c == MPI.COMM_NULL


def check_null_group(g):
    return g == MPI.GROUP_NULL


def check_null_rank(r):
    return r == MPI.PROC_NULL
