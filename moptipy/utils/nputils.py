import numpy as np


def int_range_to_dtype(min_value: int, max_value: int) -> np.dtype:
    """
    Convert an integer range to an appropriate numpy data type.
    The returned type is as compact as possible and signed types are
    preferred over unsigned types.
    The returned dtype will allow accommodating all values in the
    inclusive interval min_value..max_value.

    :param int min_value: the min_valueimum value
    :param int max_value: the max_valueimum value
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

    if (min_value >= -128) and (max_value <= 127):
        return np.dtype(np.int8)
    if (min_value >= 0) and (max_value <= 255):
        return np.dtype(np.uint8)
    if (min_value >= -32768) and (max_value <= 32767):
        return np.dtype(np.int16)
    if (min_value >= 0) and (max_value <= 65535):
        return np.dtype(np.uint16)
    if (min_value >= -2147483648) and (max_value <= 2147483647):
        return np.dtype(np.int32)
    if (min_value >= 0) and (max_value <= 4294967295):
        return np.dtype(np.uint32)
    if (min_value >= -9223372036854775808) and \
            (max_value <= 9223372036854775807):
        return np.dtype(np.int64)
    if max_value <= 18446744073709551615:
        return np.dtype(np.uint64)

    if min_value >= 0:
        raise ValueError("max_value for unsigned integers must be "
                         "less than 18446744073709551616, but is"
                         + str(max_value) + " for min_value "
                         + str(min_value) + ".")

    raise ValueError("Signed integer range cannot exceed "
                     "-9223372036854775808..9223372036854775807, "
                     "but " + str(min_value) + ".." + str(max_value)
                     + " specified.")
