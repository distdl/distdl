#ifndef __DISTDL_INTERPOLATE_H__
#define __DISTDL_INTERPOLATE_H__

#include <torch/extension.h>

#include <cmath>

#include <tuple>

// Implementation of interpolation interfaces, inspired by the
// PyTorch/ATen implementation in:
//     pytorch/aten/src/ATen/native/UpSample.h
// This is generally a re-implementation under slightly different assumptions.
// Consequently, for license compatibility, this file is licensed under the
// PyTorch license (found in distdl_root/other_licenses/PYTORCH_LICENSE)
// and not the standard DISTDL license.

void constant_interpolation_fwd_kernel_dispatch(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    double scale_factor,
    bool align_corners);

void constant_interpolation_adj_kernel_dispatch(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    double scale_factor,
    bool align_corners);

void linear_interpolation_fwd_kernel_dispatch(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    double scale_factor,
    bool align_corners);

void linear_interpolation_adj_kernel_dispatch(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    double scale_factor,
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

    // Compute the nearest-left neighbor using hte offset index provided.
    int64_t idx = static_cast<int64_t>(floorf(fac*(l_o_offset + l_o_idx) - l_i_offset));

    idx = clamp_idx_to_range(idx, 0, l_i_length-1);

    // Don't actually return the weight, it is 1.
    return idx;
}

#include <cstdio>

template<typename scalar_t>
static inline std::tuple<int64_t, scalar_t, int64_t, scalar_t> compute_linear_idx_weight(
    const int64_t& l_o_idx,
    const int64_t& l_o_offset,
    const int64_t& g_o_length,
    const int64_t& l_i_length,
    const int64_t& l_i_offset,
    const int64_t& g_i_length,
    double scale_factor,
    bool align_corners){

    // align corners changes from "counting (open ended) fence rails" to "counting fence posts"
    // Follow ATen, if there is a given scale factor, use that rather then the
    // true scale factor found using the discrete feature sizes.
    scalar_t fac = align_corners ?
                       static_cast<scalar_t>(g_i_length-1) / static_cast<scalar_t>(g_o_length-1) :
                       (scale_factor > 0 ?
                            static_cast<scalar_t>(1.0 / scale_factor) :
                            static_cast<scalar_t>(g_i_length) / static_cast<scalar_t>(g_o_length)
                        );

    // The 0.5 shift factor comes from the ATen interpolation code.  I disagree
    // with it, but to make this match the output of the torch code, we need it.
    // Also, this is why we need to clamp the index at 0 here.
    scalar_t idx_ = align_corners ?
                    fac*(l_o_offset + l_o_idx) - l_i_offset :
                    fac*(l_o_offset + l_o_idx + 0.5) - (l_i_offset + 0.5);
    idx_ = (idx_ < 0) ? scalar_t(0) : idx_;

    int64_t idx0 = static_cast<int64_t>(floorf(idx_));
    idx0 = clamp_idx_to_range(idx0, 0, l_i_length-1);
    int64_t idx1 = clamp_idx_to_range(idx0 + 1, 0, l_i_length-1);

    scalar_t lambda = idx_ - static_cast<scalar_t>(idx0);

    // Clamp to the boundaries, accounting for fp error
    if (std::abs(lambda) < 1e-4) lambda = 0.0;
    if (std::abs(lambda - 1.0) < 1e-4) lambda = 1.0;

    return std::make_tuple(idx0, 1.0 - lambda, idx1, lambda);
}

#endif
