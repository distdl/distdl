import torch

from distdl.backend import backend


class Module(torch.nn.Module):
    r"""A DistDL wrapper around PyTorch module containers.

    This class provides basic wrappers around the standard Torch base
    layer class allowing for a distributed setup forward pre-hook to
    automatically be called when necessary.

    It also defines the default DistDL back-end for all layers.

    Attributes
    ----------

    _distdl_backend :
        Default DistDL back-end used by distributed layers.

    """

    _distdl_backend = backend

    def __init__(self):

        super(Module, self).__init__()

        # Start in a non-setup state.
        self._distdl_is_setup = False

        # Register the member function that handles the layer setup
        # as a Torch pre-hook.
        self.register_forward_pre_hook(self._distdl_forward_pre_hook)

    @staticmethod
    def _distdl_forward_pre_hook(self, input):
        r"""The pre-hook for DistDL distributed layers.

        See the `PyTorch documentation <https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_forward_pre_hook>`_
        for more details on the arguments.

        Parameters
        ----------
        input :
            Tuple of inputs to the layer.
        """

        if self._distdl_module_requires_reset(input):

            if self._distdl_is_setup:
                self._distdl_module_teardown(input)

            self._distdl_module_setup(input)

        return

    def _distdl_module_setup(self, input):
        r"""Setup the DistDL distributed layer based on the input structure.

        To be defined by sub-classes if setup logic is needed.

        Parameters
        ----------
        input :
            Tuple of inputs to the layer.
        """

        pass

    def _distdl_module_teardown(self, input):
        r"""Teardown the DistDL distributed layer.

        Any resource allocated during the setup should be safely freed here.

        To be defined by sub-classes if teardown logic is needed.

        Parameters
        ----------
        input :
            Tuple of inputs to the layer.
        """

        pass

    def _distdl_input_changed(self, input):
        r"""Indicate if the input tensor has changed.

        The definition of change may be layer-specific.

        To be defined by sub-classes

        Parameters
        ----------
        input :
            Tuple of inputs to the layer.
        """

        pass

    def _distdl_module_requires_reset(self, input):
        r"""Indicate if the layer needs to be reset.

        Parameters
        ----------
        input :
            Tuple of inputs to the layer.
        """
        return not self._distdl_is_setup or self._distdl_input_changed(input)
