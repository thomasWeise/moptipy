from moptipy.api.space import Space
import numpy as np
from typing import Final

from moptipy.utils.logger import KeyValuesSection
from moptipy.utils import logging


class BitStrings(Space):
    """
    A space where each element is a bit string.
    With such a space, discrete optimization can be realized.
    """

    #: the internal type for but strings
    __DTYPE: Final = np.dtype(np.bool_)

    def __init__(self, dimension: int):
        """
        Create the vector-based search space
        :param int dimension: The dimension of the search space,
            i.e., the number of decision variables.
        """
        if (not isinstance(dimension, int)) or (dimension < 1):
            ValueError("Dimension must be positive integer, but got '"
                       + str(dimension) + "'.")
        self.dimension = dimension
        """The dimension, i.e., the number of elements of the vectors."""

    def x_create(self) -> np.ndarray:
        return np.zeros(shape=self.dimension, dtype=BitStrings.__DTYPE)

    def x_copy(self, source: np.ndarray, dest: np.ndarray):
        np.copyto(dest, source)

    def x_to_str(self, x) -> str:
        return "".join([('1' if xx else '0') for xx in x])

    def x_is_equal(self, x1, x2) -> bool:
        return np.array_equal(x1, x2)

    def x_from_str(self, text: str):
        x = self.x_create()
        x[:] = [(t == '1') for t in text]
        return x

    def get_name(self):
        return "bits" + str(self.dimension)

    def log_parameters_to(self, logger: KeyValuesSection):
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_SPACE_NUM_VARS, self.dimension)
