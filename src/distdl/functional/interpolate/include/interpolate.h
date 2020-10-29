#ifndef __DISTDL_INTERPOLATE_H__
#define __DISTDL_INTERPOLATE_H__

#include <torch/extension.h>

void constant_interp1d_fwd_kernel_interface(
    at::Tensor& output,
    const at::Tensor& input);

#endif
