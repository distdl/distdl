#include <math.h>
#include <cstdio>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include "interpolate.h"

// Implementation of nearest-left neighbor interpolation, inspired by the
// PyTorch/ATen implementation in:
//     pytorch/aten/src/ATen/native/cpu/UpSampleKernel.cpp
// This is generally a re-implementation under slightly different assumptions.
// Consequently, for license compatibility, this file is licensed under the
// PyTorch license (found in distdl_root/other_licenses/PYTORCH_LICENSE)
// and not the standard DISTDL license.

template <typename scalar_t>
void constant_interpolation_fwd_kernel_cpu(
    at::Tensor& output_,
    const at::Tensor& input_,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes
    ) {

    TORCH_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(),
                " for `output` but got dtype ", output_.dtype())

    auto input = input_.contiguous();
    auto input_data = input.data_ptr<scalar_t>();
    auto input_sizes = input.sizes().vec();

    auto output = output_.contiguous();
    auto output_data = output.data_ptr<scalar_t>();
    auto output_sizes = output.sizes().vec();

    auto ndim = input_sizes.size();

    int64_t i_nb = input_sizes[0];
    int64_t i_nc = input_sizes[1];

    // g: global
    // i/o: input/output
    // nxd: size of dth feature dimension
    // oxd: offset to dth feature dimension
    int64_t g_i_nx0, i_nx0, i_ox0, g_i_nx1, i_nx1, i_ox1, g_i_nx2, i_nx2, i_ox2;
    int64_t g_o_nx0, o_nx0, o_ox0, g_o_nx1, o_nx1, o_ox1, g_o_nx2, o_nx2, o_ox2;

    if (ndim >= 3) {
        g_i_nx0 = global_input_sizes[ndim-1];
        g_o_nx0 = global_output_sizes[ndim-1];

        i_nx0 = input_sizes[ndim-1];
        o_nx0 = output_sizes[ndim-1];

        i_ox0 = input_offsets[ndim-1];
        o_ox0 = output_offsets[ndim-1];
    }

    if (ndim >= 4) {
        g_i_nx1 = global_input_sizes[ndim-2];
        g_o_nx1 = global_output_sizes[ndim-2];

        i_nx1 = input_sizes[ndim-2];
        o_nx1 = output_sizes[ndim-2];

        i_ox1 = input_offsets[ndim-2];
        o_ox1 = output_offsets[ndim-2];
    }

    if (ndim >= 5) {
        g_i_nx2 = global_input_sizes[ndim-3];
        g_o_nx2 = global_output_sizes[ndim-3];

        i_nx2 = input_sizes[ndim-3];
        o_nx2 = output_sizes[ndim-3];

        i_ox2 = input_offsets[ndim-3];
        o_ox2 = output_offsets[ndim-3];
    }

    auto loop_1d = [&](int64_t begin, int64_t end) {

        int64_t i_x0_idx;

        for(int64_t c = begin; c < end; ++c) {
            for (int64_t o_x0_idx = 0; o_x0_idx < o_nx0; ++o_x0_idx) {

                // Compute the local input index, relative to global tensor
                // coordinates and clamp that index to the input tensor's
                // range.  If the input tensor is smaller (in the sense of
                // area) than the output tensor, this will keep accesses in
                // range.  To prevent this, ensure that they are equal area
                // or that the output is a subdomain of the area covered by
                // the input tensor.
                i_x0_idx = compute_nearest_left_idx_weight<scalar_t>(o_x0_idx, o_ox0, g_o_nx0,
                                                                     i_nx0, i_ox0, g_i_nx0);

                int64_t i_idx = compute_idx(c, i_nx0, i_x0_idx);
                int64_t o_idx = compute_idx(c, o_nx0, o_x0_idx);

                output_data[o_idx] = input_data[i_idx];
            }
        }
    };

    auto loop_2d = [&](int64_t begin, int64_t end) {

        int64_t i_x0_idx, i_x1_idx;

        for(int64_t c = begin; c < end; ++c) {
            for (int64_t o_x1_idx = 0; o_x1_idx < o_nx1; ++o_x1_idx) {
                i_x1_idx = compute_nearest_left_idx_weight<scalar_t>(o_x1_idx, o_ox1, g_o_nx1,
                                                                     i_nx1, i_ox1, g_i_nx1);

                int64_t i_c1 = compute_idx(c, i_nx1, i_x1_idx);
                int64_t o_c1 = compute_idx(c, o_nx1, o_x1_idx);

                for (int64_t o_x0_idx = 0; o_x0_idx < o_nx0; ++o_x0_idx) {

                    // Compute the local input index, relative to global tensor
                    // coordinates and clamp that index to the input tensor's
                    // range.  If the input tensor is smaller (in the sense of
                    // area) than the output tensor, this will keep accesses in
                    // range.  To prevent this, ensure that they are equal area
                    // or that the output is a subdomain of the area covered by
                    // the input tensor.
                    i_x0_idx = compute_nearest_left_idx_weight<scalar_t>(o_x0_idx, o_ox0, g_o_nx0,
                                                                         i_nx0, i_ox0, g_i_nx0);

                    int64_t i_idx = compute_idx(i_c1, i_nx0, i_x0_idx);
                    int64_t o_idx = compute_idx(o_c1, o_nx0, o_x0_idx);

                    output_data[o_idx] = input_data[i_idx];
                }
            }
        }
    };

    auto loop_3d = [&](int64_t begin, int64_t end) {

        int64_t i_x0_idx, i_x1_idx, i_x2_idx;

        for(int64_t c = begin; c < end; ++c) {
            for (int64_t o_x2_idx = 0; o_x2_idx < o_nx2; ++o_x2_idx) {
                i_x2_idx = compute_nearest_left_idx_weight<scalar_t>(o_x2_idx, o_ox2, g_o_nx2,
                                                                     i_nx2, i_ox2, g_i_nx2);

                int64_t i_c2 = compute_idx(c, i_nx2, i_x2_idx);
                int64_t o_c2 = compute_idx(c, o_nx2, o_x2_idx);

                for (int64_t o_x1_idx = 0; o_x1_idx < o_nx1; ++o_x1_idx) {
                    i_x1_idx = compute_nearest_left_idx_weight<scalar_t>(o_x1_idx, o_ox1, g_o_nx1,
                                                                         i_nx1, i_ox1, g_i_nx1);

                    int64_t i_c1 = compute_idx(i_c2, i_nx1, i_x1_idx);
                    int64_t o_c1 = compute_idx(o_c2, o_nx1, o_x1_idx);

                    for (int64_t o_x0_idx = 0; o_x0_idx < o_nx0; ++o_x0_idx) {

                        // Compute the local input index, relative to global tensor
                        // coordinates and clamp that index to the input tensor's
                        // range.  If the input tensor is smaller (in the sense of
                        // area) than the output tensor, this will keep accesses in
                        // range.  To prevent this, ensure that they are equal area
                        // or that the output is a subdomain of the area covered by
                        // the input tensor.
                        i_x0_idx = compute_nearest_left_idx_weight<scalar_t>(o_x0_idx, o_ox0, g_o_nx0,
                                                                             i_nx0, i_ox0, g_i_nx0);

                        int64_t i_idx = compute_idx(i_c1, i_nx0, i_x0_idx);
                        int64_t o_idx = compute_idx(o_c1, o_nx0, o_x0_idx);

                        output_data[o_idx] = input_data[i_idx];
                    }
                }
            }
        }
    };

    if (ndim == 3) {
        // int64_t output_slice_size = o_nx0;
        // at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE / output_slice_size / 2, loop_1d);
        at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE, loop_1d);
    }
    else if (ndim == 4) {
        // int64_t output_slice_size = o_nx1 * o_nx0;
        // at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE / output_slice_size / 4, loop_2d);
        at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE, loop_2d);
    }
    else {
        TORCH_INTERNAL_ASSERT(ndim == 5);
        // int64_t output_slice_size = o_nx2 * o_nx1 * o_nx0;
        // at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE / output_slice_size / 4, loop_2d);
        at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE, loop_3d);
    }

    if (!output_.is_contiguous()){
        output_.copy_(output);
    }
}

void constant_interpolation_fwd_kernel_dispatch(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    double scale_factor,
    bool align_corners
    ) {

    // align_corners is irrelevant for constant/nearest-left neighbor, but
    // necessary to preserve congruency of the interfaces

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "constant_interpolation_fwd", ([&]{
        constant_interpolation_fwd_kernel_cpu<scalar_t>(output,
                                                        input,
                                                        input_offsets,
                                                        global_input_sizes,
                                                        output_offsets,
                                                        global_output_sizes);
    }));
}

template <typename scalar_t>
void constant_interpolation_adj_kernel_cpu(
    at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes
    ) {

    TORCH_CHECK(grad_output_.dtype() == grad_input_.dtype(), "expected dtype ", grad_output_.dtype(),
                " for `output` but got dtype ", grad_input_.dtype())

    auto grad_input = grad_input_.contiguous();
    auto grad_input_data = grad_input.data_ptr<scalar_t>();
    auto input_sizes = grad_input.sizes().vec();

    auto grad_output = grad_output_.contiguous();
    auto grad_output_data = grad_output.data_ptr<scalar_t>();
    auto output_sizes = grad_output.sizes().vec();

    auto ndim = input_sizes.size();

    int64_t i_nb = input_sizes[0];
    int64_t i_nc = input_sizes[1];

    // g: global
    // i/o: input/output
    // nxd: size of dth feature dimension
    // oxd: offset to dth feature dimension
    int64_t g_i_nx0, i_nx0, i_ox0, g_i_nx1, i_nx1, i_ox1, g_i_nx2, i_nx2, i_ox2;
    int64_t g_o_nx0, o_nx0, o_ox0, g_o_nx1, o_nx1, o_ox1, g_o_nx2, o_nx2, o_ox2;

    if (ndim >= 3) {
        g_i_nx0 = global_input_sizes[ndim-1];
        g_o_nx0 = global_output_sizes[ndim-1];

        i_nx0 = input_sizes[ndim-1];
        o_nx0 = output_sizes[ndim-1];

        i_ox0 = input_offsets[ndim-1];
        o_ox0 = output_offsets[ndim-1];
    }

    if (ndim >= 4) {
        g_i_nx1 = global_input_sizes[ndim-2];
        g_o_nx1 = global_output_sizes[ndim-2];

        i_nx1 = input_sizes[ndim-2];
        o_nx1 = output_sizes[ndim-2];

        i_ox1 = input_offsets[ndim-2];
        o_ox1 = output_offsets[ndim-2];
    }

    if (ndim >= 5) {
        g_i_nx2 = global_input_sizes[ndim-3];
        g_o_nx2 = global_output_sizes[ndim-3];

        i_nx2 = input_sizes[ndim-3];
        o_nx2 = output_sizes[ndim-3];

        i_ox2 = input_offsets[ndim-3];
        o_ox2 = output_offsets[ndim-3];
    }

    auto loop_1d = [&](int64_t begin, int64_t end) {

        int64_t i_x0_idx;

        for(int64_t c = begin; c < end; ++c) {
            for (int64_t o_x0_idx = 0; o_x0_idx < o_nx0; ++o_x0_idx) {

                // Compute the local input index, relative to global tensor
                // coordinates and clamp that index to the input tensor's
                // range.  If the input tensor is smaller (in the sense of
                // area) than the output tensor, this will keep accesses in
                // range.  To prevent this, ensure that they are equal area
                // or that the output is a subdomain of the area covered by
                // the input tensor.
                i_x0_idx = compute_nearest_left_idx_weight<scalar_t>(o_x0_idx, o_ox0, g_o_nx0,
                                                                     i_nx0, i_ox0, g_i_nx0);

                int64_t i_idx = compute_idx(c, i_nx0, i_x0_idx);
                int64_t o_idx = compute_idx(c, o_nx0, o_x0_idx);

                grad_input_data[i_idx] += grad_output_data[o_idx];
            }
        }
    };

    auto loop_2d = [&](int64_t begin, int64_t end) {

        int64_t i_x0_idx, i_x1_idx;

        for(int64_t c = begin; c < end; ++c) {
            for (int64_t o_x1_idx = 0; o_x1_idx < o_nx1; ++o_x1_idx) {
                i_x1_idx = compute_nearest_left_idx_weight<scalar_t>(o_x1_idx, o_ox1, g_o_nx1,
                                                                     i_nx1, i_ox1, g_i_nx1);

                int64_t i_c1 = compute_idx(c, i_nx1, i_x1_idx);
                int64_t o_c1 = compute_idx(c, o_nx1, o_x1_idx);

                for (int64_t o_x0_idx = 0; o_x0_idx < o_nx0; ++o_x0_idx) {

                    // Compute the local input index, relative to global tensor
                    // coordinates and clamp that index to the input tensor's
                    // range.  If the input tensor is smaller (in the sense of
                    // area) than the output tensor, this will keep accesses in
                    // range.  To prevent this, ensure that they are equal area
                    // or that the output is a subdomain of the area covered by
                    // the input tensor.
                    i_x0_idx = compute_nearest_left_idx_weight<scalar_t>(o_x0_idx, o_ox0, g_o_nx0,
                                                                         i_nx0, i_ox0, g_i_nx0);

                    int64_t i_idx = compute_idx(i_c1, i_nx0, i_x0_idx);
                    int64_t o_idx = compute_idx(o_c1, o_nx0, o_x0_idx);

                    grad_input_data[i_idx] += grad_output_data[o_idx];
                }
            }
        }
    };

    auto loop_3d = [&](int64_t begin, int64_t end) {

        int64_t i_x0_idx, i_x1_idx, i_x2_idx;

        for(int64_t c = begin; c < end; ++c) {
            for (int64_t o_x2_idx = 0; o_x2_idx < o_nx2; ++o_x2_idx) {
                i_x2_idx = compute_nearest_left_idx_weight<scalar_t>(o_x2_idx, o_ox2, g_o_nx2,
                                                                     i_nx2, i_ox2, g_i_nx2);

                int64_t i_c2 = compute_idx(c, i_nx2, i_x2_idx);
                int64_t o_c2 = compute_idx(c, o_nx2, o_x2_idx);

                for (int64_t o_x1_idx = 0; o_x1_idx < o_nx1; ++o_x1_idx) {
                    i_x1_idx = compute_nearest_left_idx_weight<scalar_t>(o_x1_idx, o_ox1, g_o_nx1,
                                                                         i_nx1, i_ox1, g_i_nx1);

                    int64_t i_c1 = compute_idx(i_c2, i_nx1, i_x1_idx);
                    int64_t o_c1 = compute_idx(o_c2, o_nx1, o_x1_idx);

                    for (int64_t o_x0_idx = 0; o_x0_idx < o_nx0; ++o_x0_idx) {

                        // Compute the local input index, relative to global tensor
                        // coordinates and clamp that index to the input tensor's
                        // range.  If the input tensor is smaller (in the sense of
                        // area) than the output tensor, this will keep accesses in
                        // range.  To prevent this, ensure that they are equal area
                        // or that the output is a subdomain of the area covered by
                        // the input tensor.
                        i_x0_idx = compute_nearest_left_idx_weight<scalar_t>(o_x0_idx, o_ox0, g_o_nx0,
                                                                             i_nx0, i_ox0, g_i_nx0);

                        int64_t i_idx = compute_idx(i_c1, i_nx0, i_x0_idx);
                        int64_t o_idx = compute_idx(o_c1, o_nx0, o_x0_idx);

                        grad_input_data[i_idx] += grad_output_data[o_idx];
                    }
                }
            }
        }
    };

    if (ndim == 3) {
        at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE, loop_1d);
    }
    else if (ndim == 4) {
        at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE, loop_2d);
    }
    else {
        TORCH_INTERNAL_ASSERT(ndim == 5);
        at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE, loop_3d);
    }

    if (!grad_input_.is_contiguous()){
        grad_input_.copy_(grad_input);
    }
}

void constant_interpolation_adj_kernel_dispatch(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    double scale_factor,
    bool align_corners
    ) {

    // align_corners is irrelevant for constant/nearest-left neighbor, but
    // necessary to preserve congruency of the interfaces

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "constant_interpolation_adj", ([&]{
        constant_interpolation_adj_kernel_cpu<scalar_t>(grad_input,
                                                        grad_output,
                                                        input_offsets,
                                                        global_input_sizes,
                                                        output_offsets,
                                                        global_output_sizes);
    }));
}
