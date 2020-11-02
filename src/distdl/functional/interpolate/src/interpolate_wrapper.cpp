#include <torch/extension.h>

#include "interpolate.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("constant_interpolation_forward", &constant_interpolation_fwd_kernel_dispatch, "forward function");
  m.def("constant_interpolation_adjoint", &constant_interpolation_adj_kernel_dispatch, "adjoint function");
  m.def("linear_interpolation_forward", &linear_interpolation_fwd_kernel_dispatch, "forward function");
  m.def("linear_interpolation_adjoint", &linear_interpolation_adj_kernel_dispatch, "adjoint function");
}
