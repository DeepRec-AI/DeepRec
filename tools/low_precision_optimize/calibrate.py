import numpy as np


def non_linear_quant_params_search(data, bins=2048, dst_nbins=256):
    min_val, max_val = np.min(data), np.max(data)
    if min_val == max_val:
        if min_val == 0.0:
            return 0.0, 255.0
        return min(0.0, max_val), max(0.0, max_val)
    bin_width = (max_val - min_val) / bins
    hist_data = np.int32(np.minimum(np.floor((data - min_val) / bin_width), bins - 1))
    histogram = np.bincount(hist_data)

    def _get_norm(delta_begin, delta_end, density, norm_type):
        """
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        assert norm_type == "L2", "Only L2 norms are currently supported"
        norm = 0.0
        if norm_type == "L2":
            norm = (
                delta_end * delta_end * delta_end
                - delta_begin * delta_begin * delta_begin
            ) / 3
        return density * norm

    def _compute_quantization_error(next_start_bin, next_end_bin, norm_type):
        """
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = np.arange(bins)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = np.clip(src_bin_begin // dst_bin_width, 0, dst_nbins - 1)
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = np.clip(src_bin_end // dst_bin_width, 0, dst_nbins - 1)
        dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width

        density = histogram / bin_width

        norm = np.zeros(bins)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += _get_norm(delta_begin, delta_end, density, norm_type)

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * _get_norm(
            -dst_bin_width / 2, dst_bin_width / 2, density, norm_type
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += _get_norm(delta_begin, delta_end, density, norm_type)

        return np.sum(norm)

    # cumulative sum
    total = sum(histogram)
    cSum = np.cumsum(histogram, axis=0)

    stepsize = 1e-5  # granularity
    alpha = 0.0  # lower bound
    beta = 1.0  # upper bound
    start_bin = 0
    end_bin = bins - 1
    norm_min = float('inf')

    while alpha < beta:
        # Find the next step
        next_alpha = alpha + stepsize
        next_beta = beta - stepsize

        # find the left and right bins between the quantile bounds
        left = start_bin
        right = end_bin
        while left < end_bin and cSum[left] < next_alpha * total:
            left = left + 1
        while right > start_bin and cSum[right] > next_beta * total:
            right = right - 1

        # decide the next move
        next_start_bin = start_bin
        next_end_bin = end_bin
        if (left - start_bin) > (end_bin - right):
            # move the start bin
            next_start_bin = left
            alpha = next_alpha
        else:
            # move the end bin
            next_end_bin = right
            beta = next_beta

        if next_start_bin == start_bin and next_end_bin == end_bin:
            continue

        # calculate the quantization error using next_start_bin and next_end_bin
        norm = _compute_quantization_error(next_start_bin, next_end_bin, "L2")

        if norm > norm_min:
            break
        norm_min = norm
        start_bin = next_start_bin
        end_bin = next_end_bin

    new_min = min_val + bin_width * start_bin
    new_max = min_val + bin_width * (end_bin + 1)

    return new_min, new_max
