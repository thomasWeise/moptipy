from moptipy.api.space import Space
import numpy as np
from typing import Final

from moptipy.utils.logger import KeyValuesSection
from moptipy.utils import logging


class VectorSpace(Space):
    #: the character identifying the numpy data type backing the space
    KEY_NUMPY_TYPE: Final = "dtype"

    """
    A vector-based space where each element is a one-dimensional numpy array.
    """

    def __init__(self, dimension: int, dtype=np.dtype(np.float64)):
        """
        Create the vector-based search space
        :param int dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param dtype: The basic data type of the vector space,
            i.e., the type of the decision variables
        """
        if (not isinstance(dimension, int)) or (dimension < 1):
            ValueError("Dimension must be positive integer, but got '"
                       + str(dimension) + "'.")
        if not (isinstance(dtype, np.dtype) and (dtype.char in "efdgFDG")):
            ValueError("Invalid data type '" + str(dtype) + "'.")
        self.dimension = dimension
        """The dimension of the space, i.e., the vectors."""
        self.dtype = dtype
        """The basic data type of the vector elements."""

    def x_create(self) -> np.ndarray:
        return np.zeros(shape=self.dimension, dtype=self.dtype)

    def x_copy(self, source: np.ndarray, dest: np.ndarray):
        np.copyto(dest, source)

    def x_to_str(self, x) -> str:
        return ",".join([logging.format_float(xx) for xx in x])

    def x_is_equal(self, x1, x2) -> bool:
        return np.array_equal(x1, x2)

    def get_name(self):
        return "vector" + str(self.dimension) + self.dtype.char

    def log_parameters_to(self, logger: KeyValuesSection):
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_SPACE_NUM_VARS, self.dimension)
        logger.key_value(VectorSpace.KEY_NUMPY_TYPE, self.dtype.char)
