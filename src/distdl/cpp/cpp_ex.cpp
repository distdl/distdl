#include <torch/extension.h>

#include "cpp_ex.h"


int64_t get_dim(torch::Tensor x) {
  return x.dim();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_dim", &get_dim, "dim getter");
}
