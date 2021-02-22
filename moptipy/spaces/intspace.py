from moptipy.api.space import Space
import numpy as np
from typing import Final

from moptipy.utils.logger import KeyValuesSection
from moptipy.utils import logging


class IntSpace(Space):
    #: the character identifying the numpy data type backing the space
    KEY_NUMPY_TYPE: Final = "dtype"
    #: the minimum value
    KEY_MIN_VALUE: Final = "min"
    #: the maximum value
    KEY_MAX_VALUE: Final = "max"

    """
    A vector-based space where each element is a one-dimensional numpy integer array.
    """

    def __init__(self, dimension: int,
                 min_value: int,
                 max_value: int):
        """
        Create the vector-based search space
        :param int dimension: The dimension of the search space, i.e., the number of decision variables.
        :param int min_value: the minimum value
        :param int max_value: the maximum value
        """
        if (not isinstance(dimension, int)) or (dimension < 1):
            ValueError("Dimension must be positive integer, but got '"
                       + str(dimension) + "'.")
        if (not (isinstance(min_value, int) and isinstance(max_value, int))) or (min_value >= max_value):
            ValueError("min_value (" + str(min_value)
                       + ") must be an int smaller than the int max_value ("
                       + str(max_value) + ") but is not.")

        if min_value >= 0:
            if max_value <= 255:
                dtype = np.dtype(np.uint8)
            elif max_value <= 65535:
                dtype = np.dtype(np.uint16)
            elif max_value <= 4294967295:
                dtype = np.dtype(np.uint32)
            elif max_value <= 18446744073709551615:
                dtype = np.dtype(np.uint64)
            else:
                raise ValueError("max_value for unsigned integers must be less than 18446744073709551616, but is"
                                 + str(max_value))
        else:
            if (min_value >= -128) and (max_value <= 127):
                dtype = np.dtype(np.int8)
            elif (min_value >= -32768) and (max_value <= 32767):
                dtype = np.dtype(np.int16)
            elif (min_value >= -2147483648) and (max_value <= 2147483647):
                dtype = np.dtype(np.int32)
            elif (min_value >= -9223372036854775808) and (max_value <= 9223372036854775807):
                dtype = np.dtype(np.int64)
            else:
                raise ValueError("Signed integer range cannot exceed -9223372036854775808..9223372036854775807, but "
                                 + str(min_value) + ".." + str(max_value)
                                 + " specified.")

        if (not isinstance(dtype, np.dtype)) or \
           (not isinstance(dtype.char, str)) or \
           (len(dtype.char) == 1):
            ValueError("Strange error: " + str(dtype))

        self.min_value = min_value
        """The minimum permitted value."""
        self.max_value = max_value
        """The maximum permitted value."""
        self.dimension = dimension
        """The dimension of the search space, i.e., the number of decision variables."""

        self.dtype = dtype
        """The basic data type of the vector space, i.e., the type of the decision variables."""

    def x_create(self) -> np.ndarray:
        return np.zeros(shape=self.dimension, dtype=self.dtype)

    def x_copy(self, source: np.ndarray, dest: np.ndarray):
        np.copyto(dest, source)

    def x_to_str(self, x) -> str:
        return ",".join([str(xx) for xx in x])

    def x_is_equal(self, x1, x2) -> bool:
        return np.array_equal(x1, x2)

    def get_name(self):
        return "ints" + str(self.dimension) + self.dtype.char + \
               "[" + str(self.min_value) + "," + \
               str(self.max_value) + "]"

    def log_parameters_to(self, logger: KeyValuesSection):
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_SPACE_NUM_VARS, self.dimension)
        logger.key_value(IntSpace.KEY_NUMPY_TYPE, self.dtype.char)
        logger.key_value(IntSpace.KEY_MIN_VALUE, self.min_value)
        logger.key_value(IntSpace.KEY_MAX_VALUE, self.max_value)
