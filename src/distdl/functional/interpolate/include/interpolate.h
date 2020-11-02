#ifndef __DISTDL_INTERPOLATE_H__
#define __DISTDL_INTERPOLATE_H__

#include <torch/extension.h>

#include <tuple>

void constant_interpolation_fwd_kernel_dispatch(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    bool align_corners);

void constant_interpolation_adj_kernel_dispatch(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    bool align_corners);

void linear_interpolation_fwd_kernel_dispatch(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    bool align_corners);

void linear_interpolation_adj_kernel_dispatch(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    bool align_corners);

static inline int64_t clamp_idx_to_range(
    const int64_t& idx,
    const int64_t& idx_min,
    const int64_t& idx_max){

    return std::min(idx_max, std::max(idx_min, idx));
}

static inline int64_t compute_idx(
    const int64_t& j,
    const int64_t& stride,
    const int64_t& i){

    return j*stride + i;
}

template <typename scalar_t>
static inline int64_t compute_nearest_left_idx_weight(
    const int64_t& l_o_idx,
    const int64_t& l_o_offset,
    const int64_t& g_o_length,
    const int64_t& l_i_length,
    const int64_t& l_i_offset,
    const int64_t& g_i_length){

    scalar_t fac = static_cast<scalar_t>(g_i_length) / static_cast<scalar_t>(g_o_length);
    int64_t idx = static_cast<int64_t>(floorf(fac*(l_o_offset + l_o_idx) - l_i_offset));
    idx = clamp_idx_to_range(idx, 0, l_i_length-1);

    // Don't actually return the weight, it is 1.
    return idx;
}

template<typename scalar_t>
static inline std::tuple<int64_t, scalar_t, int64_t, scalar_t> compute_linear_idx_weight(
    const int64_t& l_o_idx,
    const int64_t& l_o_offset,
    const int64_t& g_o_length,
    const int64_t& l_i_length,
    const int64_t& l_i_offset,
    const int64_t& g_i_length,
    bool align_corners){

	// align corners changes from "counting (open ended) fence rails" to "counting fence posts"
    scalar_t fac = align_corners ?
                   static_cast<scalar_t>(g_i_length-1) / static_cast<scalar_t>(g_o_length-1) :
                   static_cast<scalar_t>(g_i_length) / static_cast<scalar_t>(g_o_length);
    scalar_t idx_ = fac*(l_o_offset + l_o_idx) - l_i_offset;

    int64_t idx0 = static_cast<int64_t>(floorf(idx_));
    idx0 = clamp_idx_to_range(idx0, 0, l_i_length-1);
    int64_t idx1 = clamp_idx_to_range(idx0 + 1, 0, l_i_length-1);

    scalar_t lambda = idx_ - idx0;
    // Clamp to the boundaries, accounting for fp error
    if (abs(lambda) < 1e-4)	lambda = 0.0;
    if (abs(lambda - 1.0) < 1e-4) lambda = 1.0;

    return std::make_tuple(idx0, 1.0 - lambda, idx1, lambda);
}

#endif
