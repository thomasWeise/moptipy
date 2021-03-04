"""
An implementation of a bit string based search space.
"""
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
        Create the vector-based search space
        :param int dimension: The dimension of the search space,
            i.e., the number of decision variables.
        """
        if (not isinstance(dimension, int)) or (dimension < 1):
            raise ValueError("Dimension must be positive integer, but got '"
                             + str(dimension) + "'.")
        self.dimension = dimension
        """The dimension, i.e., the number of elements of the vectors."""

    def create(self) -> np.ndarray:
        """
        This method creates a bit string filled with False
        :return: the string

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
        np.copyto(dest, source)

    def to_str(self, x: np.ndarray) -> str:
        return "".join([('1' if xx else '0') for xx in x])

    def is_equal(self, x1: np.ndarray, x2: np.ndarray) -> bool:
        return np.array_equal(x1, x2)

    def from_str(self, text: str) -> np.ndarray:
        x = self.create()
        x[:] = [(t == '1') for t in text]
        return x

    def validate(self, x: np.ndarray) -> None:
        if not (isinstance(x, np.ndarray)):
            raise ValueError("x must be an numpy.ndarray, but is a '"
                             + str(type(x)) + ".")
        if x.dtype != BitStrings.__DTYPE:
            raise ValueError("x must be of type '" + str(BitStrings.__DTYPE)
                             + "' but is of type '" + str(x.dtype) + "'.")
        if (len(x.shape) != 1) or (x.shape[0] != self.dimension):
            raise ValueError("x must be of shape (" + str(self.dimension)
                             + ") but is of shape " + str(x.shape) + ".")

    def scale(self) -> int:
        return 2 ** self.dimension

    def get_name(self) -> str:
        return "bits" + str(self.dimension)

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_SPACE_NUM_VARS, self.dimension)
