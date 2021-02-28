from moptipy.api.space import Space
import numpy as np
from typing import Final

from moptipy.utils.logger import KeyValuesSection
from moptipy.utils import logging
from moptipy.utils.nputils import int_range_to_dtype


class IntSpace(Space):
    """
    A space where each element is a one-dimensional numpy integer array.
    Such spaces can serve as basis for implementing combinatorial
    optimization and can be extended to host permutations.
    """

    #: the character identifying the numpy data type backing the space
    KEY_NUMPY_TYPE: Final = "dtype"
    #: the minimum value
    KEY_MIN: Final = "min"
    #: the maximum value
    KEY_MAX: Final = "max"

    def __init__(self, dimension: int,
                 min_value: int,
                 max_value: int):
        """
        Create the integer-based search space
        :param int dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param int min_value: the minimum value
        :param int max_value: the maximum value
        """
        if (not isinstance(dimension, int)) or (dimension < 1):
            raise ValueError("Dimension must be positive integer, but got '"
                             + str(dimension) + "'.")

        self.dtype = int_range_to_dtype(min_value=min_value,
                                        max_value=max_value)
        """The basic data type of the vector space elements."""

        if (not isinstance(self.dtype, np.dtype)) or \
                (not isinstance(self.dtype.char, str)) or \
                (len(self.dtype.char) != 1):
            raise ValueError("Strange error: " + str(self.dtype))

        self.min_value = min_value
        """The minimum permitted value."""
        self.max_value = max_value
        """The maximum permitted value."""
        self.dimension = dimension
        """The dimension, i.e., the number of elements of the vectors."""

    def create(self) -> np.ndarray:
        """
        This method creates a integer vector filled with the minimal value.
        :return: the vector

        >>> from moptipy.spaces.intspace import IntSpace
        >>> s = IntSpace(dimension=12, min_value=5, max_value=332)
        >>> v = s.create()
        >>> print(s.to_str(v))
        5,5,5,5,5,5,5,5,5,5,5,5
        >>> print(v.dtype)
        int16
        """
        return np.full(shape=self.dimension,
                       fill_value=self.min_value,
                       dtype=self.dtype)

    def copy(self, source: np.ndarray, dest: np.ndarray):
        np.copyto(dest, source)

    def to_str(self, x: np.ndarray) -> str:
        return ",".join([str(xx) for xx in x])

    def is_equal(self, x1: np.ndarray, x2: np.ndarray) -> bool:
        return np.array_equal(x1, x2)

    def from_str(self, text: str) -> np.ndarray:
        x = np.fromstring(text, dtype=self.dtype, sep=",")
        if len(x) != self.dimension:
            raise ValueError("'" + text + "' does not have dimension "
                             + str(self.dimension))
        for xx in x:
            if (xx < self.min_value) or (xx > self.max_value):
                ValueError("Value '" + str(xx) + "' in '" + text
                           + "' outside of range '" + str(self.min_value)
                           + ".." + str(self.max_value))
        return x

    def validate(self, x: np.ndarray):
        if not (isinstance(x, np.ndarray)):
            raise ValueError("x must be an numpy.ndarray, but is a '"
                             + str(type(x)) + ".")
        if x.dtype != self.dtype:
            raise ValueError("x must be of type '" + str(self.dtype)
                             + "' but is of type '" + str(x.dtype) + "'.")
        if (len(x.shape) != 1) or (x.shape[0] != self.dimension):
            raise ValueError("x must be of shape (" + str(self.dimension)
                             + ") but is of shape " + str(x.shape) + ".")
        if not all(x >= self.min_value):
            raise ValueError("All elements of x must be >= "
                             + str(self.min_value) + ", but "
                             + str(x.min()) + " was encountered.")
        if not all(x <= self.max_value):
            raise ValueError("All elements of x must be <= "
                             + str(self.max_value) + ", but "
                             + str(x.max()) + " was encountered.")

    def scale(self) -> int:
        return (self.max_value - self.min_value + 1) ** self.dimension

    def get_name(self) -> str:
        return "ints" + str(self.dimension) + self.dtype.char \
               + str(self.min_value) + "-" + str(self.max_value)

    def log_parameters_to(self, logger: KeyValuesSection):
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_SPACE_NUM_VARS, self.dimension)
        logger.key_value(IntSpace.KEY_NUMPY_TYPE, self.dtype.char)
        logger.key_value(IntSpace.KEY_MIN, self.min_value)
        logger.key_value(IntSpace.KEY_MAX, self.max_value)
