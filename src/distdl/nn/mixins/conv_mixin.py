class ConvMixin:
    r"""A mixin providing general support for distributed convolution layers.

    """

    def _compute_min_input_range(self,
                                 idx,
                                 kernel_size,
                                 stride,
                                 padding,
                                 dilation):
        r"""Compute left index required to apply a kernel at a given index.

        .. warning::
           This does not currently take stride and dilation into account.
           Therefore, the padding values may not be correct in these cases.

        Parameters
        ----------
        idx :
            Cartesian index that is the reference location for the kernel.
        kernel_size:
            Size of the kernel in all indexed dimensions.
        stride:
            Size of the stride in all indexed dimensions.
        padding:
            Size of any specified 0 padding in any indexed dimensions.
        dilation:
            Size of the dilation parameter in all indexed dimensions.

        Returns
        -------
        Left-most tensor index needed to apply a kernel with the given
        parameters.

        """

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
        r"""Compute right index required to apply a kernel at a given index.

        .. warning::
           This does not currently take stride and dilation into account.
           Therefore, the padding values may not be correct in these cases.

        Parameters
        ----------
        idx :
            Cartesian index that is the reference location for the kernel.
        kernel_size:
            Size of the kernel in all indexed dimensions.
        stride:
            Size of the stride in all indexed dimensions.
        padding:
            Size of any specified 0 padding in any indexed dimensions.
        dilation:
            Size of the dilation parameter in all indexed dimensions.

        Returns
        -------
        Right-most tensor index needed to apply a kernel with the given
        parameters.

        """

        # incorrect, does not take stride and dilation into account
        # padding might also not be correct in these cases...
        kernel_offsets = (kernel_size - 1) / 2

        bases = idx + kernel_offsets - padding
        return bases + kernel_offsets
