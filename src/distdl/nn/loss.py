import torch
import numpy as np

from distdl.functional import ZeroVolumeCorrectorFunction
from distdl.nn.module import Module
from distdl.nn.sum_reduce import SumReduce
from distdl.utilities.torch import TensorStructure


class DistributedLossBase(Module):

    r"""
    Base class for distributed loss functions.

    Parameters
    ----------
    P_x : Partition
        Partition of input and target tensors.
    reduction : str, optional
        Reduction mode.  Default: "mean".

    Attributes
    ----------
    BaseLossLayer : type
        The PyTorch loss layer this layer is equivalent to.
    _valid_reductions : list
        The PyTorch reduction modes that are allowed for this loss.

    Warning
    -------
    Weight functions are not yet supported.

    """
    
    BaseLossLayer = None
    _valid_reductions = ["none", "mean", "sum"]

    def __init__(self, P_x, reduction="mean"):
        super(DistributedLossBase, self).__init__()

        if reduction in self._valid_reductions:
            self.reduction = reduction
            if self.reduction == "none":
                self.local_loss_function = self.BaseLossLayer(reduction="none")
            else:
                # For "sum", "mean", and "batchmean" we need to get local sums first,
                # but we have to normalize *after* the distributed reduction.
                self.local_loss_function = self.BaseLossLayer(reduction="sum")
        else:
            raise ValueError(f"Invalid reduction mode for Loss type {self.__class__}.")

        self.P_x = P_x

        P_0_base = P_x.create_partition_inclusive([0])
        self.P_0 = P_0_base.create_cartesian_topology_partition([1]*P_x.dim)

        # Clean up temporary resources.
        P_0_base.deactivate()

        # Define the layer handling the reduction of the loss
        self.sum_reduce = SumReduce(self.P_x, self.P_0)

        # By default, no normalization is required
        self.normalization_factor = 1


    def _distdl_module_setup(self, input):
        r"""Distributed loss setup function.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        if not self.P_x.active:
            return

        # If the reduction mode is "mean", we need the total size of the
        # global input tensor.
        if self.reduction == "mean":
            N_local = np.prod(input[0].shape)
            self.normalization_factor = int(self.P_x.allreduce_data(np.asarray(N_local)))

        self._distdl_is_setup = True
        self._input_tensor_structure = TensorStructure(input[0])

    def _distdl_module_teardown(self, input):
        r"""Distributed loss teardown function.

        Nullifies the necessary partition functions.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        self.normalization_factor = 1

        # Reset any info about the input
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()


    def _distdl_input_changed(self, input):
        r"""Determine if the structure of inputs has changed.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        new_tensor_structure = TensorStructure(input[0])

        return self._input_tensor_structure != new_tensor_structure


    def forward(self, local_input, local_target):
        r"""Distributed loss forward function.

        Computes the loss function in parallel by calling the underlying
        PyTorch loss function and following the PyTorch reduction specification
        on that loss.

        If the reduction mode is "none" the raw loss values are returned on
        each worker. If it is any of the other reduction modes, the reduced
        loss value is returned on the 0th rank of `P_x` only.  All other
        ranks return zero-volume tensors _through the zero-volume corrector_.
        This is because the distributed sum-reduce only returns the value on
        the root rank and returns a zero-volume tensor elsewhere. However,
        PyTorch cannot backpropagate from a loss returning zero-volume. Thus,
        we use the zero-volume corrector to return a fake value.

        Note, this is equivalent to all-reducing the loss in the forward call
        and not communicating in backward call, because the adjoint of a
        sum-reduce is a broadcast and an all-reduce is equivalent to a
        summ-reduce followed by a broadcast.

        Parameters
        ----------
        local_input :
            Current worker's portion of the input tensor.
        local_target :
            Current worker's portion of the target tensor.

        """

        if not self.P_x.active:
            return input.clone()

        # Evaluate the local loss
        local_loss = self.local_loss_function(local_input, local_target)

        # If the reduction is none, each rank keeps its local part
        if self.reduction == "none":
            return local_loss

        # Torch losses have no shape in the output, but the sum-reduction
        # layer requires the tensor to have shape.
        local_loss = torch.reshape(local_loss, (1,))
        global_loss = self.sum_reduce(local_loss)

        # The [0] is so that we return a scalar wrapped in a tensor, as
        # the torch loss functions do.
        if self.P_0.active:
            global_loss = global_loss[0] / self.normalization_factor

        return ZeroVolumeCorrectorFunction.apply(global_loss)


class DistributedL1Loss(DistributedLossBase):
    r"""
    Distributed L1 loss.  See PyTorch documentation for details.
    """
    BaseLossLayer = torch.nn.L1Loss


class DistributedMSELoss(DistributedLossBase):
    r"""
    Distributed MSE loss.  See PyTorch documentation for details.
    """
    BaseLossLayer = torch.nn.MSELoss


class DistributedPoissonNLLLoss(DistributedLossBase):
    r"""
    Distributed Poisson negative log-liklihood loss.  See PyTorch documentation for details.
    """
    BaseLossLayer = torch.nn.PoissonNLLLoss


class DistributedBCELoss(DistributedLossBase):
    r"""
    Distributed binary cross-entropy loss.  See PyTorch documentation for details.
    """
    BaseLossLayer = torch.nn.BCELoss


class DistributedBCEWithLogitsLoss(DistributedLossBase):
    r"""
    Distributed BCE with logits loss.  See PyTorch documentation for details.
    """
    BaseLossLayer = torch.nn.BCEWithLogitsLoss


class DistributedKLDivLoss(DistributedLossBase):
    r"""
    Distributed KL Divergence loss.  See PyTorch documentation for details.
    """
    BaseLossLayer = torch.nn.KLDivLoss

    _valid_reductions = ["none", "mean", "sum", "batchmean"]

    def _distdl_module_setup(self, input):
        r"""Distributed KL Divergence loss setup function.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        if not self.P_x.active:
            return

        # If the reduction mode is "mean", we need the total size of the
        # global input tensor.
        if self.reduction == "mean":
            N_local = np.prod(input[0].shape)
            self.normalization_factor = int(self.P_x.allreduce_data(np.asarray(N_local)))
        # If the reduction mode is "batchmean", we need the batch size, which
        # means we only add the shapes along the batch axis.
        elif self.reduction == "batchmean":
            if all(coord == 0 for coord in self.P_x.index[1:]):
                N_local = input[0].shape[0]
            else:
                N_local = 0
            self.normalization_factor = int(self.P_x.allreduce_data(np.asarray(N_local)))

        self._distdl_is_setup = True
        self._input_tensor_structure = TensorStructure(input[0])

