import warnings

from distdl.nn.repartition import Repartition

warnings.warn("DistributedTranspose is deprecated; use Repartition", DeprecationWarning)

DistributedTranspose = Repartition
