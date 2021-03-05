"""An implementation of a bit string based search space."""
from typing import Final

import numpy as np

from moptipy.api.space import Space
from moptipy.utils import logging
from moptipy.utils.logger import KeyValueSection


class BitStrings(Space):
    """
    A space where each element is a bit string.

    With such a space, discrete optimization can be realized.
    """

    #: the internal type for but strings
    __DTYPE: Final = np.dtype(np.bool_)

    def __init__(self, dimension: int) -> None:
        """
        Create the vector-based search space.

        :param int dimension: The dimension of the search space,
            i.e., the number of decision variables.
        """
        if not isinstance(dimension, int):
            raise TypeError("dimension must be integer, but got '"
                            + str(type(dimension)) + "'.")
        if (dimension < 1) or (dimension > 1_000_000_000):
            raise ValueError("dimension must be in 1..1_000_000_000, but got "
                             + str(dimension) + ".")
        self.dimension = dimension
        """The dimension, i.e., the number of elements of the vectors."""

    def create(self) -> np.ndarray:
        """
        Create a bit string filled with `False`.

        :return: the string
        :rtype: np.ndarray

        >>> from moptipy.spaces.bitstrings import BitStrings
        >>> s = BitStrings(8)
        >>> v = s.create()
        >>> print(s.to_str(v))
        00000000
        >>> print(v.dtype)
        bool
        """
        return np.zeros(shape=self.dimension, dtype=BitStrings.__DTYPE)

    def copy(self, source: np.ndarray, dest: np.ndarray) -> None:
        """
        Copy the contents of one bit string to another.

        :param np.ndarray source: the source string
        :param np.ndarray dest: the target string
        """
        np.copyto(dest, source)

    def to_str(self, x: np.ndarray) -> str:
        """
        Convert a bit string to a string, using `0` and `1` without separator.

        :param np.ndarray x: the bit string
        :return: the string
        :rtype: str
        """
        return "".join([('1' if xx else '0') for xx in x])

    def is_equal(self, x1: np.ndarray, x2: np.ndarray) -> bool:
        """
        Check if two bit strings are equal.

        :param np.ndarray x1: the first bit string
        :param np.ndarray x2: the second
        :return: `True` if the contents of both bit strings are equal,
            `False` otherwise
        :rtype: bool
        """
        return np.array_equal(x1, x2)

    def from_str(self, text: str) -> np.ndarray:
        """
        Convert a string to a bit string.

        :param str text: the text
        :return: the vector
        :raises TypeError: if `text` is not a `str`
        :raises ValueError: if `text` cannot be converted to a valid vector
        """
        if not (isinstance(text, str)):
            raise TypeError("text must be str, but is "
                            + str(type(text)) + ".")
        x = self.create()
        x[:] = [(t == '1') for t in text]
        self.validate(x)
        return x

    def validate(self, x: np.ndarray) -> None:
        """
        Validate a bit string.

        :param np.ndarray x: the bit string
        :rtype: np.ndarray
        :raises TypeError: if the string is not an element of this space.
        :raises ValueError: if the shape of the string
        """
        if not (isinstance(x, np.ndarray)):
            raise TypeError("x must be an numpy.ndarray, but is a '"
                            + str(type(x)) + ".")
        if x.dtype != BitStrings.__DTYPE:
            raise TypeError("x must be of type '" + str(BitStrings.__DTYPE)
                            + "' but is of type '" + str(x.dtype) + "'.")
        if (len(x.shape) != 1) or (x.shape[0] != self.dimension):
            raise ValueError("x must be of shape (" + str(self.dimension)
                             + ") but is of shape " + str(x.shape) + ".")

    def scale(self) -> int:
        """
        Get the scale of the bit string space.

        :return: 2 ** dimension
        :rtype: int
        """
        return 1 << (self.dimension - 1)  # = 2 ** self.dimension

    def get_name(self) -> str:
        """
        Get the name of this space.

        :return: "bits" + dimension
        :rtype: str
        """
        return "bits" + str(self.dimension)

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param KeyValueLogger logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_SPACE_NUM_VARS, self.dimension)
