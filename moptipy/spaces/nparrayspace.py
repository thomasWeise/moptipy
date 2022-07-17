"""The base class for spaces based on numpy arrays."""
from typing import Final

import numpy

from moptipy.api.logging import KEY_SPACE_NUM_VARS
from moptipy.api.space import Space
from moptipy.utils.logger import CSV_SEPARATOR
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import KEY_NUMPY_TYPE, val_numpy_type, array_to_str
from moptipy.utils.types import type_error


class NPArraySpace(Space):
    """
    A space where each element is a one-dimensional numpy array.

    Such spaces can serve as basis for implementing combinatorial
    optimization and can be extended to host permutations.
    """

    def __init__(self, dimension: int, dtype: numpy.dtype) -> None:
        """
        Create the numpy array-based search space.

        :param dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param dtype: the data type
        """
        if not isinstance(dimension, int):
            raise type_error(dimension, "dimension", int)
        if (dimension < 1) or (dimension > 1_000_000_000):
            raise ValueError("dimension must be in 1..1_000_000_000, "
                             f"but got {dimension}.")

        if not isinstance(dtype, numpy.dtype):
            raise type_error(dtype, "dtype", numpy.dtype)
        if (not isinstance(dtype.char, str)) or (len(dtype.char) != 1):
            raise ValueError(
                f"dtype.char must be str of length 1, but is {dtype.char}")

        #: The basic data type of the vector space elements.
        self.dtype: Final[numpy.dtype] = dtype
        #: The dimension, i.e., the number of elements of the vectors.
        self.dimension: Final[int] = dimension
        # the function forwards
        self.copy = numpy.copyto  # type: ignore
        self.is_equal = numpy.array_equal  # type: ignore
        self.to_str = array_to_str  # type: ignore

    def create(self) -> numpy.ndarray:
        """
        Create a vector with all zeros.

        :return: the vector
        """
        return numpy.zeros(shape=self.dimension, dtype=self.dtype)

    def from_str(self, text: str) -> numpy.ndarray:
        """
        Convert a string to a vector.

        :param text: the text
        :return: the vector
        :raises TypeError: if `text` is not a `str`
        :raises ValueError: if `text` cannot be converted to a valid vector
        """
        if not (isinstance(text, str)):
            raise type_error(text, "text", str)
        x = numpy.fromstring(text, dtype=self.dtype, sep=CSV_SEPARATOR)
        self.validate(x)
        return x

    def validate(self, x: numpy.ndarray) -> None:
        """
        Validate a numpy nd-array.

        :param x: the numpy vector
        :raises TypeError: if the string is not an element of this space.
        :raises ValueError: if the shape of the vector is wrong or any of its
            element is not finite.
        """
        if not isinstance(x, numpy.ndarray):
            raise type_error(x, "x", numpy.ndarray)
        if x.dtype != self.dtype:
            raise ValueError(
                f"x must be of type {self.dtype} but is of type {x.dtype}.")
        if (len(x.shape) != 1) or (x.shape[0] != self.dimension):
            raise ValueError(f"x must be of shape ({self.dimension}) but is "
                             f"of shape {x.shape}.")

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_SPACE_NUM_VARS, self.dimension)
        logger.key_value(KEY_NUMPY_TYPE, val_numpy_type(self.dtype))
