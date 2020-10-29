#include <torch/extension.h>

#include "interpolate.h"

#include <cstdio>

template <typename scalar_t>
void constant_interp1d_fwd_kernel(
    at::Tensor& output_,
    const at::Tensor& input_) {

    TORCH_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(),
                " for `output` but got strype ", output_.dtype())

    auto input = input_.contiguous();
    auto input_data = input.data_ptr<scalar_t>();
    auto input_sizes = input.sizes().vec();

    auto output = output_.contiguous();
    auto output_data = output.data_ptr<scalar_t>();
    auto output_sizes = output.sizes().vec();

    auto ndim = input_sizes.size();

    printf("OK %s \n", input_.dtype().name());

}




void constant_interp1d_fwd_kernel_interface(
    at::Tensor& output,
    const at::Tensor& input) {

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "constant_interp1d_fwd", ([&]{
        constant_interp1d_fwd_kernel<scalar_t>(output, input);
    }));
}
