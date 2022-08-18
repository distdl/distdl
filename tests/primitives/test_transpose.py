def test_transpose_is_repartition():

    from distdl.nn.repartition import Repartition
    from distdl.nn.transpose import DistributedTranspose

    assert DistributedTranspose is Repartition
