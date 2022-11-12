"""A nullary operator filling a vector with uniformly distribute values."""

from math import isfinite
from typing import Final

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op0
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error


class Op0Uniform(Op0):
    """Fill a vector with uniformly distributed random values."""

    def __init__(self, x_min: float, x_max: float) -> None:
        """
        Initialize the operator.

        :param x_min: the minimum value
        :param x_max: the maximum value
        """
        super().__init__()
        if not isinstance(x_min, float):
            raise type_error(x_min, "x_min", float)
        if not isfinite(x_min):
            raise ValueError(f"x_min must be finite, but is {x_min}.")
        if not isinstance(x_max, float):
            raise type_error(x_max, "x_max", float)
        if not isfinite(x_max):
            raise ValueError(f"x_max must be finite, but is {x_max}.")
        if x_min >= x_max:
            raise ValueError(f"x_max > x_min must hold, but got "
                             f"x_min={x_min} and x_max={x_max}.")
        #: the minimum value for each element of the vectors in the space
        self.x_min: Final[float] = x_min
        #: the maximum value for each element of the vectors in the space
        self.x_max: Final[float] = x_max

    def __str__(self) -> str:
        """
        Get the name of this operator.

        :returns: the name of this space
        """
        return "uniform"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("xMin", self.x_min, also_hex=True)
        logger.key_value("xMax", self.x_max, also_hex=True)

    def op0(self, random: Generator, dest: np.ndarray) -> None:
        """
        Fill the string `dest` with random values.

        :param random: the random number generator
        :param dest: the vector to be filled. Afterwards it contains uniformly
            distributed random values
        """
        np.copyto(dest, random.uniform(self.x_min, self.x_max, len(dest)))
