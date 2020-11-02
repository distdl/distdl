#include <math.h>
#include <cstdio>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include "interpolate.h"

template <typename scalar_t>
void linear_interpolation_fwd_kernel_cpu(
    at::Tensor& output_,
    const at::Tensor& input_,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    bool align_corners
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

        int64_t i_x0_idx0, i_x0_idx1;
        scalar_t w0_0, w0_1;

        for(int64_t c = begin; c < end; ++c) {
            for (int64_t o_x0_idx = 0; o_x0_idx < o_nx0; ++o_x0_idx) {

                // Compute the local input index, relative to global tensor
                // coordinates and clamp that index to the input tensor's
                // range.  If the input tensor is smaller (in the sense of
                // area) than the output tensor, this will keep accesses in
                // range.  To prevent this, ensure that they are equal area
                // or that the output is a subdomain of the area covered by
                // the input tensor.
                std::tie(i_x0_idx0, w0_0, i_x0_idx1, w0_1) =
                    compute_linear_idx_weight<scalar_t>(o_x0_idx, o_ox0, g_o_nx0,
                                                        i_nx0, i_ox0, g_i_nx0,
                                                        align_corners);

                int64_t o_idx_0 = compute_idx(c, o_nx0, o_x0_idx);

                output_data[o_idx_0]  = w0_0*input_data[compute_idx(c, i_nx0, i_x0_idx0)];
                output_data[o_idx_0] += w0_1*input_data[compute_idx(c, i_nx0, i_x0_idx1)];
            }
        }
    };

    auto loop_2d = [&](int64_t begin, int64_t end) {

        int64_t i_x0_idx0, i_x0_idx1;
        int64_t i_x1_idx0, i_x1_idx1;

        // last dimension is fast
        scalar_t w0_0, w0_1;
        scalar_t w1_0, w1_1;

        for(int64_t c = begin; c < end; ++c) {

            for (int64_t o_x1_idx = 0; o_x1_idx < o_nx1; ++o_x1_idx) {
                std::tie(i_x1_idx0, w1_0, i_x1_idx1, w1_1) =
                    compute_linear_idx_weight<scalar_t>(o_x1_idx, o_ox1, g_o_nx1,
                                                        i_nx1, i_ox1, g_i_nx1,
                                                        align_corners);

                // 0/1 is x1 index (+0 or +1); X means unset dimension
                int64_t i_idx_0X = compute_idx(c, i_nx1, i_x1_idx0);
                int64_t i_idx_1X = compute_idx(c, i_nx1, i_x1_idx1);

                int64_t o_idx_0X = compute_idx(c, o_nx1, o_x1_idx);

                for (int64_t o_x0_idx = 0; o_x0_idx < o_nx0; ++o_x0_idx) {

                    // Compute the local input index, relative to global tensor
                    // coordinates and clamp that index to the input tensor's
                    // range.  If the input tensor is smaller (in the sense of
                    // area) than the output tensor, this will keep accesses in
                    // range.  To prevent this, ensure that they are equal area
                    // or that the output is a subdomain of the area covered by
                    // the input tensor.
                    std::tie(i_x0_idx0, w0_0, i_x0_idx1, w0_1) =
                        compute_linear_idx_weight<scalar_t>(o_x0_idx, o_ox0, g_o_nx0,
                                                            i_nx0, i_ox0, g_i_nx0,
                                                            align_corners);

                    int64_t o_idx_00 = compute_idx(o_idx_0X, o_nx0, o_x0_idx);

                    output_data[o_idx_00]  = w1_0*w0_0*input_data[compute_idx(i_idx_0X, i_nx0, i_x0_idx0)];
                    output_data[o_idx_00] += w1_0*w0_1*input_data[compute_idx(i_idx_0X, i_nx0, i_x0_idx1)];
                    output_data[o_idx_00] += w1_1*w0_0*input_data[compute_idx(i_idx_1X, i_nx0, i_x0_idx0)];
                    output_data[o_idx_00] += w1_1*w0_1*input_data[compute_idx(i_idx_1X, i_nx0, i_x0_idx1)];
                }
            }
        }
    };

    auto loop_3d = [&](int64_t begin, int64_t end) {

        int64_t i_x0_idx0, i_x0_idx1;
        int64_t i_x1_idx0, i_x1_idx1;
        int64_t i_x2_idx0, i_x2_idx1;

        // last dimension is fast
        scalar_t w0_0, w0_1;
        scalar_t w1_0, w1_1;
        scalar_t w2_0, w2_1;

        // c = batch-channel index
        for(int64_t c = begin; c < end; ++c) {
            for (int64_t o_x2_idx = 0; o_x2_idx < o_nx2; ++o_x2_idx) {
                std::tie(i_x2_idx0, w2_0, i_x2_idx1, w2_1) =
                    compute_linear_idx_weight<scalar_t>(o_x2_idx, o_ox2, g_o_nx2,
                                                        i_nx2, i_ox2, g_i_nx2,
                                                        align_corners);

                // 0/1 is x2 index (+0 or +1); X means unset dimension
                int64_t i_idx_0XX = compute_idx(c, i_nx2, i_x2_idx0);
                int64_t i_idx_1XX = compute_idx(c, i_nx2, i_x2_idx1);

                int64_t o_idx_0XX = compute_idx(c, o_nx2, o_x2_idx);

                for (int64_t o_x1_idx = 0; o_x1_idx < o_nx1; ++o_x1_idx) {
                    std::tie(i_x1_idx0, w1_0, i_x1_idx1, w1_1) =
                        compute_linear_idx_weight<scalar_t>(o_x1_idx, o_ox1, g_o_nx1,
                                                            i_nx1, i_ox1, g_i_nx1,
                                                            align_corners);

                    // 0/1 is x2,x1 index (+0 or +1); X means unset dimension
                    int64_t i_idx_00X = compute_idx(i_idx_0XX, i_nx1, i_x1_idx0);
                    int64_t i_idx_01X = compute_idx(i_idx_0XX, i_nx1, i_x1_idx1);
                    int64_t i_idx_10X = compute_idx(i_idx_1XX, i_nx1, i_x1_idx0);
                    int64_t i_idx_11X = compute_idx(i_idx_1XX, i_nx1, i_x1_idx1);

                    int64_t o_idx_00X = compute_idx(o_idx_0XX, o_nx1, o_x1_idx);

                    for (int64_t o_x0_idx = 0; o_x0_idx < o_nx0; ++o_x0_idx) {

                        // Compute the local input index, relative to global tensor
                        // coordinates and clamp that index to the input tensor's
                        // range.  If the input tensor is smaller (in the sense of
                        // area) than the output tensor, this will keep accesses in
                        // range.  To prevent this, ensure that they are equal area
                        // or that the output is a subdomain of the area covered by
                        // the input tensor.
                        std::tie(i_x0_idx0, w0_0, i_x0_idx1, w0_1) =
                            compute_linear_idx_weight<scalar_t>(o_x0_idx, o_ox0, g_o_nx0,
                                                                i_nx0, i_ox0, g_i_nx0,
                                                                align_corners);

                        int64_t o_idx_000 = compute_idx(o_idx_00X, o_nx0, o_x0_idx);

                        output_data[o_idx_000]  = w2_0*w1_0*w0_0*input_data[compute_idx(i_idx_00X, i_nx0, i_x0_idx0)];
                        output_data[o_idx_000] += w2_0*w1_0*w0_1*input_data[compute_idx(i_idx_00X, i_nx0, i_x0_idx1)];
                        output_data[o_idx_000] += w2_0*w1_1*w0_0*input_data[compute_idx(i_idx_01X, i_nx0, i_x0_idx0)];
                        output_data[o_idx_000] += w2_0*w1_1*w0_1*input_data[compute_idx(i_idx_01X, i_nx0, i_x0_idx1)];
                        output_data[o_idx_000] += w2_1*w1_0*w0_0*input_data[compute_idx(i_idx_10X, i_nx0, i_x0_idx0)];
                        output_data[o_idx_000] += w2_1*w1_0*w0_1*input_data[compute_idx(i_idx_10X, i_nx0, i_x0_idx1)];
                        output_data[o_idx_000] += w2_1*w1_1*w0_0*input_data[compute_idx(i_idx_11X, i_nx0, i_x0_idx0)];
                        output_data[o_idx_000] += w2_1*w1_1*w0_1*input_data[compute_idx(i_idx_11X, i_nx0, i_x0_idx1)];
                    }
                }
            }
        }
    };

    if (ndim == 3) {
        int64_t output_slice_size = o_nx0;
        at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE / output_slice_size / 2, loop_1d);
    }
    else if (ndim == 4) {
        int64_t output_slice_size = o_nx1 * o_nx0;
        at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE / output_slice_size / 4, loop_2d);
    }
    else {
        TORCH_INTERNAL_ASSERT(ndim == 5);
        int64_t output_slice_size = o_nx2 * o_nx1 * o_nx0;
        at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE / output_slice_size / 8, loop_3d);
    }

    if (!output_.is_contiguous()){
        output_.copy_(output);
    }
}

void linear_interpolation_fwd_kernel_dispatch(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    bool align_corners
    ) {

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "linear_interpolation_fwd", ([&]{
        linear_interpolation_fwd_kernel_cpu<scalar_t>(output,
                                                      input,
                                                      input_offsets,
                                                      global_input_sizes,
                                                      output_offsets,
                                                      global_output_sizes,
                                                      align_corners);
    }));
}

template <typename scalar_t>
void linear_interpolation_adj_kernel_cpu(
    at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    bool align_corners
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

        int64_t i_x0_idx0, i_x0_idx1;
        scalar_t w0_0, w0_1;

        for(int64_t c = begin; c < end; ++c) {
            for (int64_t o_x0_idx = 0; o_x0_idx < o_nx0; ++o_x0_idx) {

                // Compute the local input index, relative to global tensor
                // coordinates and clamp that index to the input tensor's
                // range.  If the input tensor is smaller (in the sense of
                // area) than the output tensor, this will keep accesses in
                // range.  To prevent this, ensure that they are equal area
                // or that the output is a subdomain of the area covered by
                // the input tensor.
                std::tie(i_x0_idx0, w0_0, i_x0_idx1, w0_1) =
                    compute_linear_idx_weight<scalar_t>(o_x0_idx, o_ox0, g_o_nx0,
                                                        i_nx0, i_ox0, g_i_nx0,
                                                        align_corners);

                int64_t o_idx_0 = compute_idx(c, o_nx0, o_x0_idx);

                grad_input_data[compute_idx(c, i_nx0, i_x0_idx0)] += w0_0*grad_output_data[o_idx_0];
                grad_input_data[compute_idx(c, i_nx0, i_x0_idx1)] += w0_1*grad_output_data[o_idx_0];
            }
        }
    };

    auto loop_2d = [&](int64_t begin, int64_t end) {

        int64_t i_x0_idx0, i_x0_idx1;
        int64_t i_x1_idx0, i_x1_idx1;

        // last dimension is fast
        scalar_t w0_0, w0_1;
        scalar_t w1_0, w1_1;

        for(int64_t c = begin; c < end; ++c) {
            for (int64_t o_x1_idx = 0; o_x1_idx < o_nx1; ++o_x1_idx) {
                std::tie(i_x1_idx0, w1_0, i_x1_idx1, w1_1) =
                    compute_linear_idx_weight<scalar_t>(o_x1_idx, o_ox1, g_o_nx1,
                                                        i_nx1, i_ox1, g_i_nx1,
                                                        align_corners);

                // 0/1 is x1 index (+0 or +1); X means unset dimension
                int64_t i_idx_0X = compute_idx(c, i_nx1, i_x1_idx0);
                int64_t i_idx_1X = compute_idx(c, i_nx1, i_x1_idx1);

                int64_t o_idx_0X = compute_idx(c, o_nx1, o_x1_idx);

                for (int64_t o_x0_idx = 0; o_x0_idx < o_nx0; ++o_x0_idx) {

                    // Compute the local input index, relative to global tensor
                    // coordinates and clamp that index to the input tensor's
                    // range.  If the input tensor is smaller (in the sense of
                    // area) than the output tensor, this will keep accesses in
                    // range.  To prevent this, ensure that they are equal area
                    // or that the output is a subdomain of the area covered by
                    // the input tensor.
                    std::tie(i_x0_idx0, w0_0, i_x0_idx1, w0_1) =
                        compute_linear_idx_weight<scalar_t>(o_x0_idx, o_ox0, g_o_nx0,
                                                            i_nx0, i_ox0, g_i_nx0,
                                                            align_corners);

                    int64_t i_idx_00 = compute_idx(o_idx_0X, o_nx0, o_x0_idx);

                    grad_input_data[compute_idx(i_idx_0X, i_nx0, i_x0_idx0)] += w1_0*w0_0*grad_output_data[i_idx_00];
                    grad_input_data[compute_idx(i_idx_0X, i_nx0, i_x0_idx1)] += w1_0*w0_1*grad_output_data[i_idx_00];
                    grad_input_data[compute_idx(i_idx_1X, i_nx0, i_x0_idx0)] += w1_1*w0_0*grad_output_data[i_idx_00];
                    grad_input_data[compute_idx(i_idx_1X, i_nx0, i_x0_idx1)] += w1_1*w0_1*grad_output_data[i_idx_00];
                }
            }
        }
    };

    auto loop_3d = [&](int64_t begin, int64_t end) {

        int64_t i_x0_idx0, i_x0_idx1;
        int64_t i_x1_idx0, i_x1_idx1;
        int64_t i_x2_idx0, i_x2_idx1;

        // last dimension is fast
        scalar_t w0_0, w0_1;
        scalar_t w1_0, w1_1;
        scalar_t w2_0, w2_1;

        // c = batch-channel index
        for(int64_t c = begin; c < end; ++c) {
            for (int64_t o_x2_idx = 0; o_x2_idx < o_nx2; ++o_x2_idx) {
                std::tie(i_x2_idx0, w2_0, i_x2_idx1, w2_1) =
                    compute_linear_idx_weight<scalar_t>(o_x2_idx, o_ox2, g_o_nx2,
                                                        i_nx2, i_ox2, g_i_nx2,
                                                        align_corners);

                // 0/1 is x2 index (+0 or +1); X means unset dimension
                int64_t i_idx_0XX = compute_idx(c, i_nx2, i_x2_idx0);
                int64_t i_idx_1XX = compute_idx(c, i_nx2, i_x2_idx1);

                int64_t o_idx_0XX = compute_idx(c, o_nx2, o_x2_idx);

                for (int64_t o_x1_idx = 0; o_x1_idx < o_nx1; ++o_x1_idx) {
                    std::tie(i_x1_idx0, w1_0, i_x1_idx1, w1_1) =
                        compute_linear_idx_weight<scalar_t>(o_x1_idx, o_ox1, g_o_nx1,
                                                            i_nx1, i_ox1, g_i_nx1,
                                                            align_corners);

                    // 0/1 is x2,x1 index (+0 or +1); X means unset dimension
                    int64_t i_idx_00X = compute_idx(i_idx_0XX, i_nx1, i_x1_idx0);
                    int64_t i_idx_01X = compute_idx(i_idx_0XX, i_nx1, i_x1_idx1);
                    int64_t i_idx_10X = compute_idx(i_idx_1XX, i_nx1, i_x1_idx0);
                    int64_t i_idx_11X = compute_idx(i_idx_1XX, i_nx1, i_x1_idx1);

                    int64_t o_idx_00X = compute_idx(o_idx_0XX, o_nx1, o_x1_idx);

                    for (int64_t o_x0_idx = 0; o_x0_idx < o_nx0; ++o_x0_idx) {

                        // Compute the local input index, relative to global tensor
                        // coordinates and clamp that index to the input tensor's
                        // range.  If the input tensor is smaller (in the sense of
                        // area) than the output tensor, this will keep accesses in
                        // range.  To prevent this, ensure that they are equal area
                        // or that the output is a subdomain of the area covered by
                        // the input tensor.
                        std::tie(i_x0_idx0, w0_0, i_x0_idx1, w0_1) =
                            compute_linear_idx_weight<scalar_t>(o_x0_idx, o_ox0, g_o_nx0,
                                                                i_nx0, i_ox0, g_i_nx0,
                                                                align_corners);

                        int64_t o_idx_000 = compute_idx(o_idx_00X, o_nx0, o_x0_idx);

                        grad_input_data[compute_idx(i_idx_00X, i_nx0, i_x0_idx0)] += w2_0*w1_0*w0_0*grad_output_data[o_idx_000];
                        grad_input_data[compute_idx(i_idx_00X, i_nx0, i_x0_idx1)] += w2_0*w1_0*w0_1*grad_output_data[o_idx_000];
                        grad_input_data[compute_idx(i_idx_01X, i_nx0, i_x0_idx0)] += w2_0*w1_1*w0_0*grad_output_data[o_idx_000];
                        grad_input_data[compute_idx(i_idx_01X, i_nx0, i_x0_idx1)] += w2_0*w1_1*w0_1*grad_output_data[o_idx_000];
                        grad_input_data[compute_idx(i_idx_10X, i_nx0, i_x0_idx0)] += w2_1*w1_0*w0_0*grad_output_data[o_idx_000];
                        grad_input_data[compute_idx(i_idx_10X, i_nx0, i_x0_idx1)] += w2_1*w1_0*w0_1*grad_output_data[o_idx_000];
                        grad_input_data[compute_idx(i_idx_11X, i_nx0, i_x0_idx0)] += w2_1*w1_1*w0_0*grad_output_data[o_idx_000];
                        grad_input_data[compute_idx(i_idx_11X, i_nx0, i_x0_idx1)] += w2_1*w1_1*w0_1*grad_output_data[o_idx_000];
                    }
                }
            }
        }
    };

    if (ndim == 3) {
        int64_t output_slice_size = o_nx0;
        at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE / output_slice_size / 2, loop_1d);
    }
    else if (ndim == 4) {
        int64_t output_slice_size = o_nx1 * o_nx0;
        at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE / output_slice_size / 4, loop_2d);
    }
    else {
        TORCH_INTERNAL_ASSERT(ndim == 5);
        int64_t output_slice_size = o_nx2 * o_nx1 * o_nx0;
        at::parallel_for(0, i_nb*i_nc, at::internal::GRAIN_SIZE / output_slice_size / 8, loop_3d);
    }

    if (!grad_input_.is_contiguous()){
        grad_input_.copy_(grad_input);
    }
}

void linear_interpolation_adj_kernel_dispatch(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef input_offsets,
    at::IntArrayRef global_input_sizes,
    at::IntArrayRef output_offsets,
    at::IntArrayRef global_output_sizes,
    bool align_corners
    ) {

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "linear_interpolation_adj", ([&]{
        linear_interpolation_adj_kernel_cpu<scalar_t>(grad_input,
                                                      grad_output,
                                                      input_offsets,
                                                      global_input_sizes,
                                                      output_offsets,
                                                      global_output_sizes,
                                                      align_corners);
    }));
}
