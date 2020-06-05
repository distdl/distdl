import pytest


@pytest.mark.mpi
@pytest.fixture(scope="function")
def barrier_fence_fixture():
    from mpi4py import MPI

    MPI.COMM_WORLD.Barrier()

    yield

    MPI.COMM_WORLD.Barrier()


@pytest.mark.mpi
@pytest.fixture(scope="function")
def comm_split_fixture(request):
    from mpi4py import MPI

    min_size = request.param

    # Isolate the required number of processors
    if MPI.COMM_WORLD.rank < min_size:
        color = 0
        base_comm = MPI.COMM_WORLD.Split(color)
        return base_comm, True
    else:
        color = 1
        base_comm = MPI.COMM_WORLD.Split(color)
        return base_comm, False
