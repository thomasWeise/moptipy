from moptipy.api.space import Space
import numpy as np
from typing import Final

from moptipy.utils.logger import KeyValueSection
from moptipy.utils import logging


class VectorSpace(Space):
    """
    A vector-based space where each element is a one-dimensional numpy array.
    Such spaces are useful for continuous optimization.
    """

    #: the character identifying the numpy data type backing the space
    KEY_NUMPY_TYPE: Final = "dtype"

    def __init__(self, dimension: int, dtype=np.dtype(np.float64)):
        """
        Create the vector-based search space
        :param int dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param dtype: The basic data type of the vector space,
            i.e., the type of the decision variables
        """
        if (not isinstance(dimension, int)) or (dimension < 1):
            raise ValueError("Dimension must be positive integer, but got '"
                             + str(dimension) + "'.")
        if not (isinstance(dtype, np.dtype) and (dtype.char in "efdgFDG")):
            raise ValueError("Invalid data type '" + str(dtype) + "'.")
        self.dimension = dimension
        """The dimension of the space, i.e., the vectors."""
        self.dtype = dtype
        """The basic data type of the vector elements."""

    def create(self) -> np.ndarray:
        return np.zeros(shape=self.dimension, dtype=self.dtype)

    def copy(self, source: np.ndarray, dest: np.ndarray):
        np.copyto(dest, source)

    def to_str(self, x: np.ndarray) -> str:
        return ",".join([logging.format_float(xx) for xx in x])

    def is_equal(self, x1: np.ndarray, x2: np.ndarray) -> bool:
        return np.array_equal(x1, x2)

    def from_str(self, text: str) -> np.ndarray:
        x = np.fromstring(text, dtype=self.dtype, sep=",")
        if len(x) != self.dimension:
            raise ValueError("'" + text + "' does not have dimension "
                             + str(self.dimension))
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
        if any(np.isnan(x)):
            raise ValueError("No element must be NaN.")

    def scale(self) -> int:
        if self.dtype.char == "e":
            exponent = 5
            mantissa = 10
            is_complex = False
        elif self.dtype.char == "f":
            exponent = 8
            mantissa = 23
            is_complex = False
        elif self.dtype == "d":
            exponent = 11
            mantissa = 52
            is_complex = False
        elif self.dtype == "g":
            exponent = 15
            mantissa = 112
            is_complex = False
        elif self.dtype == "F":
            exponent = 8
            mantissa = 23
            is_complex = True
        elif self.dtype == "D":
            exponent = 11
            mantissa = 52
            is_complex = True
        elif self.dtype == "G":
            exponent = 15
            mantissa = 112
            is_complex = True
        else:
            raise ValueError("Invalid dtype " + str(self.dtype))

        base = 2 * ((2 ** exponent) - 1) * (2 ** mantissa) - 1
        if is_complex:
            base = base * base
        return base ** self.dimension

    def get_name(self) -> str:
        return "vector" + str(self.dimension) + self.dtype.char

    def log_parameters_to(self, logger: KeyValueSection):
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_SPACE_NUM_VARS, self.dimension)
        logger.key_value(VectorSpace.KEY_NUMPY_TYPE, self.dtype.char)
