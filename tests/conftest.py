import pytest


@pytest.mark.mpi
@pytest.fixture(scope="function")
def barrier_fence_fixture():
    from mpi4py import MPI

    MPI.COMM_WORLD.Barrier()

    yield

    MPI.COMM_WORLD.Barrier()
