import numpy as np
from typing import Final

__NP_INTS: Final = \
    tuple([(t[0], int(t[1].min), int(t[1].max))
           for t in ((tt, np.iinfo(tt))
                     for tt in (np.dtype(ttt)
                                for ttt in [np.int8, np.uint8,
                                            np.int16, np.uint16,
                                            np.int32, np.uint32,
                                            np.int64, np.uint64]))])

DEFAULT_INT: Final = (__NP_INTS[len(__NP_INTS) - 2])[0]

__NP_INT_MAX: Final = {}
for i in __NP_INTS:
    __NP_INT_MAX[i[0]] = i[2]
__NP_INT_MIN: Final = {}
for i in __NP_INTS:
    __NP_INT_MIN[i[0]] = i[1]


def int_range_to_dtype(min_value: int, max_value: int) -> np.dtype:
    """
    Convert an integer range to an appropriate numpy data type.
    The returned type is as compact as possible and signed types are
    preferred over unsigned types.
    The returned dtype will allow accommodating all values in the
    inclusive interval min_value..max_value.

    :param int min_value: the minimum value
    :param int max_value: the maximum value
    :return np.dtype: the numpy integer range
    """

    if not isinstance(min_value, int):
        raise ValueError("min_value must be int, but is '"
                         + str(type(min_value)) + "'.")
    if not isinstance(max_value, int):
        raise ValueError("max_value must be int, but is '"
                         + str(type(max_value)) + "'.")
    if min_value > max_value:
        raise ValueError("min_value must be <>>= max_value, but min_value="
                         + str(min_value) + " and max_value="
                         + str(max_value) + " was provided.")

    for t in __NP_INTS:
        if (min_value >= t[1]) and (max_value <= t[2]):
            return t[0]

    ll = len(__NP_INTS) - 1
    if min_value >= 0:
        raise ValueError("max_value for unsigned integers must be "
                         "<=" + str((__NP_INTS[ll])[2]) + ", but is "
                         + str(max_value) + " for min_value "
                         + str(min_value) + ".")

    ll = ll - 1
    raise ValueError("Signed integer range cannot exceed "
                     + str((__NP_INTS[ll])[1]) + ".." + str((__NP_INTS[ll])[2])
                     + ", but " + str(min_value) + ".." + str(max_value)
                     + " was specified.")


def intmax(shape, dtype: np.dtype = DEFAULT_INT) -> np.ndarray:
    """
    Create an integer array of the given length filled with the maximum value
    :param shape: the requested shape
    :param dtype: the data type (defaults to 64 bit integers)
    :return: the new array
    """
    return np.full(shape=shape,
                   fill_value=__NP_INT_MAX[dtype],
                   dtype=dtype)


def intmin(shape, dtype: np.dtype = DEFAULT_INT) -> np.ndarray:
    """
    Create an integer array of the given length filled with the minimum value
    :param shape: the requested shape
    :param dtype: the data type (defaults to 64 bit integers)
    :return: the new array
    """
    return np.full(shape=shape,
                   fill_value=__NP_INT_MIN[dtype],
                   dtype=dtype)
