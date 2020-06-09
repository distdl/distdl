import torch

from distdl.backend import backend


class Module(torch.nn.Module):

    _distdl_backend = backend

    def __init__(self):

        super(Module, self).__init__()

        self._distdl_is_setup = False
        self.register_forward_pre_hook(self._distdl_forward_pre_hook)

    @staticmethod
    def _distdl_forward_pre_hook(self, input):

        if self._distdl_module_requires_reset():

            if self._distdl_is_setup:
                self._distdl_module_teardown()

            self._distdl_module_setup()

        return

    def _distdl_module_setup(self):
        pass

    def _distdl_module_teardown(self):
        pass

    def _distdl_module_requires_reset(self):
        return (not self._distdl_is_setup) or True
