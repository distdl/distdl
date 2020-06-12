class PoolingMixin:

    def _compute_min_input_range(self,
                                 idx,
                                 kernel_size,
                                 stride,
                                 padding,
                                 dilation):

        # incorrect, does not take dilation and padding into account
        return stride * idx + 0

    def _compute_max_input_range(self,
                                 idx,
                                 kernel_size,
                                 stride,
                                 padding,
                                 dilation):

        # incorrect, does not take dilation and padding into account
        return stride * idx + kernel_size - 1
