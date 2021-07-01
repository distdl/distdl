import torch
import numpy as np

from distdl.nn.module import Module
from distdl.nn.sum_reduce import SumReduce
from distdl.utilities.torch import TensorStructure
from distdl.utilities.torch import zero_volume_tensor

class ZeroVolumeCorrectorFunction(torch.autograd.Function):

    r"""Functional implementation of a wrapper for distributed loss outputs.

    Implements the required `forward()` and adjoint (`backward()`) operations
    in a consistent way that keeps PyTorch happy.

    Generally, PyTorch cannot backpropagate through zero-volume tensors, which
    are the default output when a distributed layer has no meaningful output.
    Thus, when an operation is at the end of a network, and a worker needs to
    return a zero-volume tensor, we actually have to return a valid tensor.  Yet,
    on the backward pass we must ignore any input and return the adjoint of the
    zero-volume output, which is a zero-volume grad input.

    For example, distributed loss functions assume that the loss value is
    reduced to rank 0.  Thus, rank 0 can return the actual loss but all other
    ranks would return nothing.  However, to satisfy PyTorch and Autograd's
    need for a scalar function in the root of the backward call, all other
    ranks must a tiny tensor from the forward call.  The backward
    call "reverses" this behavior, where rank 0 preserves the provided grad
    input but all other ranks inject the expected zero-volume tensor.

    DistDL distributed functions can handle zero-volume inputs to both forward
    and adjoint calls without this extra step.

    """

    @staticmethod
    def forward(ctx, input):

        r"""Forward function of zero-volume corrector wrapper.

        Parameters
        ----------
        ctx :
            PyTorch context.
        input : `torch.tensor`
            Input tensor.

        Returns
        -------
        output :
            `input` if it is not zero-volume, a tensor with a single element if the input is zero-volume.

        """

        ctx.sh = input.shape
        ctx.zero_volume = np.prod(input.shape) == 0

        if ctx.zero_volume:
            return torch.tensor(0.0, requires_grad=True).float()
        else:
            return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        r"""Adjoint function of zero-volume corrector wrapper.

        This method interfaces to the adjoint of the Jacobian of the
        forward  zero-volume corrector operation.

        Parameters
        ----------
        ctx :
            PyTorch context.
        grad_output : `torch.tensor`
            Input tensor.

        Returns
        -------
        output :
            `grad_output` if `input` was not zero-volume, a zero-volume tensor otherwise.

        """

        sh = ctx.sh

        if ctx.zero_volume:
            return zero_volume_tensor(sh[0])
        else:
            return grad_output.clone()


class _DistributedLoss(Module):

    _valid_reductions = ["none", "mean", "sum"]

    def __init__(self, P_x, reduction="mean"):
        super(_DistributedLoss, self).__init__()

        if reduction in self._valid_reductions:
            self.reduction = reduction
            if self.reduction == "none":
                self.local_loss_function = self.BaseLossLayer(reduction="none")
            else:
                # for "sum", "mean", and "batchmean" we need to get local sums first
                self.local_loss_function = self.BaseLossLayer(reduction="sum")
        else:
            raise ValueError(f"Invalid reduction mode for Loss type {self.__class__}.")

        self.P_x = P_x

        P_0_base = P_x.create_partition_inclusive([0])
        self.P_0 = P_0_base.create_cartesian_topology_partition([1]*P_x.dim)

        P_0_base.deactivate()

        self.sum_reduce = SumReduce(self.P_x, self.P_0)

        self.normalization_factor = 1


    def _distdl_module_setup(self, input):
        r"""Broadcast module setup function.

        Constructs the necessary partition functions to implement the above
        described broadcast pattern.  This function performs collective
        communication across the input and output partitions.

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

        if self.reduction == "mean":
            N_local = np.prod(input[0].shape)
            self.normalization_factor = int(self.P_x.allreduce_data(np.asarray(N_local)))

        self._distdl_is_setup = True
        self._input_tensor_structure = TensorStructure(input[0])

    def _distdl_module_teardown(self, input):
        r"""Broadcast module teardown function.

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

        if not self.P_x.active:
            return input.clone()

        local_loss = self.local_loss_function(local_input, local_target)

        if self.reduction == "none":
            # If the reduction is none, each rank keeps its local part
            return local_loss

        local_loss = torch.reshape(local_loss, (1,))
        global_loss = self.sum_reduce(local_loss)

        if self.P_0.active:
            # The [0] is so that we return a scalar wrapped in a tensor, as
            # the torch loss functions do.
            global_loss = global_loss[0] / self.normalization_factor

        return ZeroVolumeCorrectorFunction.apply(global_loss)


class DistributedL1Loss(_DistributedLoss):
    BaseLossLayer = torch.nn.L1Loss

class DistributedMSELoss(_DistributedLoss):
    BaseLossLayer = torch.nn.MSELoss

class DistributedPoissonNLLLoss(_DistributedLoss):
    BaseLossLayer = torch.nn.PoissonNLLLoss

# class DistributedGaussianNLLLoss(_DistributedLoss):
#     BaseLossLayer = torch.nn.GaussianNLLLoss

class DistributedBCELoss(_DistributedLoss):
    BaseLossLayer = torch.nn.BCELoss

class DistributedBCEWithLogitsLoss(_DistributedLoss):
    BaseLossLayer = torch.nn.BCEWithLogitsLoss

class DistributedKLDivLoss(_DistributedLoss):
    BaseLossLayer = torch.nn.KLDivLoss

    _valid_reductions = ["none", "mean", "sum", "batchmean"]

    def _distdl_module_setup(self, input):
        r"""Broadcast module setup function.

        Constructs the necessary partition functions to implement the above
        described broadcast pattern.  This function performs collective
        communication across the input and output partitions.

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

        if self.reduction == "mean":
            N_local = np.prod(input[0].shape)
            self.normalization_factor = int(self.P_x.allreduce_data(np.asarray(N_local)))
        elif self.reduction == "batchmean":
            # We need the total batch size, which means we only add the shapes along the batch axis
            if all(coord == 0 for coord in self.P_x.index[1:]):
                N_local = input[0].shape[0]
            else:
                N_local = 0
            self.normalization_factor = int(self.P_x.allreduce_data(np.asarray(N_local)))

        self._distdl_is_setup = True
        self._input_tensor_structure = TensorStructure(input[0])

