"""The internal base class for spaces based on numpy arrays."""
from typing import Final

import numpy as np

from moptipy.api.logging import KEY_SPACE_NUM_VARS
from moptipy.api.space import Space
from moptipy.utils.logger import KeyValueSection
from moptipy.utils.nputils import KEY_NUMPY_TYPE


class _NPArraySpace(Space):
    """
    A space where each element is a one-dimensional numpy array.

    Such spaces can serve as basis for implementing combinatorial
    optimization and can be extended to host permutations.
    """

    def __init__(self, dimension: int, dtype: np.dtype) -> None:
        """
        Create the numpy array-based search space.

        :param int dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param np.dtype dtype: the data type
        """
        if not isinstance(dimension, int):
            raise TypeError(
                f"dimension must be integer, but got {type(dimension)}.")
        if (dimension < 1) or (dimension > 1_000_000_000):
            raise ValueError("dimension must be in 1..1_000_000_000, "
                             f"but got {dimension}.")

        if not isinstance(dtype, np.dtype):
            raise TypeError(f"dtype must be np.dtype, but is {type(dtype)}.")
        if (not isinstance(dtype.char, str)) or (len(dtype.char) != 1):
            raise ValueError(
                f"dtype.char must be str of length 1, but is {dtype.char}")

        #: The basic data type of the vector space elements.
        self.dtype: Final[np.dtype] = dtype
        #: The dimension, i.e., the number of elements of the vectors.
        self.dimension: Final[int] = dimension
        # the function forwards
        self.copy = np.copyto  # type: ignore
        self.is_equal = np.array_equal  # type: ignore

    def create(self) -> np.ndarray:
        """
        Create a vector with all zeros.

        :return: the vector
        :rtype: np.ndarray
        """
        return np.zeros(shape=self.dimension, dtype=self.dtype)

    def to_str(self, x: np.ndarray) -> str:
        """
        Convert an integer string to a string, using `,` as separator.

        :param np.ndarray x: the integer string
        :return: the string
        :rtype: str
        """
        return ",".join([str(xx) for xx in x])

    def from_str(self, text: str) -> np.ndarray:
        """
        Convert a string to an integer string.

        :param str text: the text
        :return: the vector
        :rtype: np.ndarray
        :raises TypeError: if `text` is not a `str`
        :raises ValueError: if `text` cannot be converted to a valid vector
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, but is {type(text)}.")
        x = np.fromstring(text, dtype=self.dtype, sep=",")
        self.validate(x)
        return x

    def validate(self, x: np.ndarray) -> None:
        """
        Validate a string.

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

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param KeyValueLogger logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_SPACE_NUM_VARS, self.dimension)
        logger.key_value(KEY_NUMPY_TYPE, self.dtype.char)
