"""An implementation of an integer string based search space."""
from typing import Final

import numpy as np

from moptipy.api.space import Space
from moptipy.api import logging
from moptipy.utils.logger import KeyValueSection
from moptipy.utils.nputils import int_range_to_dtype

#: the character identifying the numpy data type backing the space
KEY_NUMPY_TYPE: Final[str] = "dtype"
#: the minimum value
KEY_MIN: Final[str] = "min"
#: the maximum value
KEY_MAX: Final[str] = "max"


class IntSpace(Space):
    """
    A space where each element is a one-dimensional numpy integer array.

    Such spaces can serve as basis for implementing combinatorial
    optimization and can be extended to host permutations.
    """

    def __init__(self, dimension: int,
                 min_value: int,
                 max_value: int) -> None:
        """
        Create the integer-based search space.

        :param int dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param int min_value: the minimum value
        :param int max_value: the maximum value
        """
        if not isinstance(dimension, int):
            raise TypeError(
                f"dimension must be integer, but got {type(dimension)}.")
        if (dimension < 1) or (dimension > 1_000_000_000):
            raise ValueError("dimension must be in 1..1_000_000_000, "
                             f"but got {dimension}.")

        #: The basic data type of the vector space elements.
        self.dtype: Final[np.dtype] = \
            int_range_to_dtype(min_value=min_value, max_value=max_value)

        if (not isinstance(self.dtype, np.dtype)) or \
                (not isinstance(self.dtype.char, str)) or \
                (len(self.dtype.char) != 1):
            raise ValueError(f"Strange error: {self.dtype} found.")

        #: The minimum permitted value.
        self.min_value: Final[int] = min_value

        #: The maximum permitted value.
        self.max_value: Final[int] = max_value

        #: The dimension, i.e., the number of elements of the vectors.
        self.dimension: Final[int] = dimension

    def create(self) -> np.ndarray:
        """
        Create a integer vector filled with the minimal value.

        :return: the vector
        :rtype: np.ndarray

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

    def copy(self, source: np.ndarray, dest: np.ndarray) -> None:
        """
        Copy the contents of one integer string to another.

        :param np.ndarray source: the source string
        :param np.ndarray dest: the target string
        """
        np.copyto(dest, source)

    def to_str(self, x: np.ndarray) -> str:
        """
        Convert an integer string to a string, using `,` as separator.

        :param np.ndarray x: the integer string
        :return: the string
        :rtype: str
        """
        return ",".join([str(xx) for xx in x])

    def is_equal(self, x1: np.ndarray, x2: np.ndarray) -> bool:
        """
        Check if two integer vectors are equal.

        :param np.ndarray x1: the first int vector
        :param np.ndarray x2: the second
        :return: `True` if the contents of both int vectors are equal,
            `False` otherwise
        :rtype: bool
        """
        return np.array_equal(x1, x2)

    def from_str(self, text: str) -> np.ndarray:
        """
        Convert a string to an integer string.

        :param str text: the text
        :return: the vector
        :rtype: np.ndarray
        :raises TypeError: if `text` is not a `str`
        :raises ValueError: if `text` cannot be converted to a valid vector
        """
        if not (isinstance(text, str)):
            raise TypeError(f"text must be str, but is {type(text)}.")
        x = np.fromstring(text, dtype=self.dtype, sep=",")
        self.validate(x)
        return x

    def validate(self, x: np.ndarray) -> None:
        """
        Validate an integer string.

        :param np.ndarray x: the integer string
        :raises TypeError: if the string is not an element of this space.
        :raises ValueError: if the shape of the vector is wrong or any of its
            element is not finite.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"x must be an numpy.ndarray, but is a {type(x)}.")
        if x.dtype != self.dtype:
            raise TypeError(
                f"x must be of type {self.dtype} but is of type {x.dtype}.")
        if (len(x.shape) != 1) or (x.shape[0] != self.dimension):
            raise ValueError(f"x must be of shape ({self.dimension}) but is "
                             f"of shape {x.shape}.")
        if not all(x >= self.min_value):
            raise ValueError(
                f"All elements of x must be >= {self.min_value}, but "
                f"{x.min()} was encountered.")
        if not all(x <= self.max_value):
            raise ValueError(
                f"All elements of x must be <= {self.max_value}, but "
                f"{x.max()} was encountered.")

    def scale(self) -> int:
        """
        Get the number of possible different integer strings.

        :return: (max_value - min_value + 1) ** dimension
        :rtype: int
        """
        return (self.max_value - self.min_value + 1) ** self.dimension

    def get_name(self) -> str:
        """
        Get the name of this integer space.

        :return: "ints" + dimension + dtype.char + min_value + "-" + max_value
        :rtype: int
        """
        return f"ints{self.dimension}{self.dtype.char}" \
               f"{self.min_value}-{self.max_value}"

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param KeyValueLogger logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_SPACE_NUM_VARS, self.dimension)
        logger.key_value(KEY_NUMPY_TYPE, self.dtype.char)
        logger.key_value(KEY_MIN, self.min_value)
        logger.key_value(KEY_MAX, self.max_value)
