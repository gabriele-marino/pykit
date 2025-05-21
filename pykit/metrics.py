import numpy as np
from numpy.typing import NDArray

from pykit.type_hints import ArrayLike


def _to_float64_array(data: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Input must be a 1D array-like of floats.")
    return array


def average_ci_width(lower: ArrayLike, upper: ArrayLike) -> float:
    lower_arr = _to_float64_array(lower)
    upper_arr = _to_float64_array(upper)
    if lower_arr.shape != upper_arr.shape:
        raise ValueError("lower and upper must have the same shape.")
    widths = upper_arr - lower_arr
    return float(np.mean(widths))


def coverage(lower: ArrayLike, upper: ArrayLike, true_values: ArrayLike) -> float:
    lower_arr = _to_float64_array(lower)
    upper_arr = _to_float64_array(upper)
    true_arr = _to_float64_array(true_values)
    if not (len(lower_arr) == len(upper_arr) == len(true_arr)):
        raise ValueError("All input arrays must have the same length.")
    contained = (lower_arr <= true_arr) & (true_arr <= upper_arr)
    return float(np.mean(contained))


def coverage_efficiency(
    lower: ArrayLike, upper: ArrayLike, true_values: ArrayLike
) -> float:
    avg_width = average_ci_width(lower, upper)
    if not avg_width:
        raise ValueError("Average width is zero, cannot compute efficiency.")
    cov = coverage(lower, upper, true_values)
    return cov / avg_width
