class ConvMixin:

    def _compute_min_input_range(self,
                                 idx,
                                 kernel_size,
                                 stride,
                                 padding,
                                 dilation):

        # incorrect, does not take stride and dilation into account
        # padding might also not be correct in these cases...
        kernel_offsets = (kernel_size - 1) / 2

        # for even sized kernels, always shortchange the left side
        kernel_offsets[kernel_size % 2 == 0] -= 1

        bases = idx + kernel_offsets - padding
        return bases - kernel_offsets

    def _compute_max_input_range(self,
                                 idx,
                                 kernel_size,
                                 stride,
                                 padding,
                                 dilation):

        # incorrect, does not take stride and dilation into account
        # padding might also not be correct in these cases...
        kernel_offsets = (kernel_size - 1) / 2

        bases = idx + kernel_offsets - padding
        return bases + kernel_offsets
