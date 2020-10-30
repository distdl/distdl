#ifndef __DISTDL_INTERPOLATE_H__
#define __DISTDL_INTERPOLATE_H__

#include <torch/extension.h>

void constant_interpolation_fwd_kernel_dispatch(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes);

void constant_interpolation_adj_kernel_dispatch(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes);

template <typename scalar_t>
static inline int64_t compute_nearest_left_idx(
    const int64_t& o_idx,
    const int64_t& i_length,
    const int64_t& i_offset,
    const int64_t& o_length,
    const int64_t& o_offset){

    scalar_t fac = static_cast<scalar_t>(i_length) / static_cast<scalar_t>(o_length);

    return static_cast<int64_t>(floorf(fac*(o_offset + o_idx) - i_offset));
}

static inline int64_t clamp_idx_to_range(
    const int64_t& idx,
    const int64_t& idx_min,
    const int64_t& idx_max){

    return std::min(idx_max, std::max(idx_min, idx));
}

static inline int64_t compute_idx(
    const int64_t& j,
    const int64_t& ni,
    const int64_t& i){

    return j*ni + i;
}

#endif
