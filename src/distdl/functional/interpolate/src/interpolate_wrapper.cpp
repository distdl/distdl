#include <torch/extension.h>

#include "interpolate.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &constant_interp1d_fwd_kernel_interface, "forward function");
}
