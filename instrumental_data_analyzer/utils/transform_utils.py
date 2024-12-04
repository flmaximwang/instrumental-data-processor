import numpy as np

def rescale(data, old_start, old_end, new_start, new_end):
    """
    Rescale data with linear transformation, old start will be mapped to new start, old end will be mapped to new end
    """
    return (data - old_start) / (old_end - old_start) * (new_end - new_start) + new_start

def rescale_to_0_1(data, old_start, old_end):
    """
    Rescale data with linear transformation, old start will be mapped to 0, old end will be mapped to 1
    """
    return rescale(data, old_start, old_end, 0, 1)

def extend_range(range, ratio):
    """
    Extend the range by ratio
    """
    center = (range[0] + range[1]) / 2
    return (center - ratio * (center - range[0]), center + ratio * (range[1] - center))