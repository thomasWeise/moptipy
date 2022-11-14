"""A nullary operator filling a vector with uniformly distribute values."""

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op0
from moptipy.utils.bounds import FloatBounds
from moptipy.utils.logger import KeyValueLogSection


class Op0Uniform(Op0, FloatBounds):
    """Fill a vector with uniformly distributed random values."""

    def __init__(self, min_value: float = -1.0,
                 max_value: float = 1.0) -> None:
        """
        Initialize the operator.

        :param min_value: the minimum value
        :param max_value: the maximum value
        """
        Op0.__init__(self)
        FloatBounds.__init__(self, min_value, max_value)

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
        Op0.log_parameters_to(self, logger)
        FloatBounds.log_parameters_to(self, logger)

    def op0(self, random: Generator, dest: np.ndarray) -> None:
        """
        Fill the string `dest` with random values.

        :param random: the random number generator
        :param dest: the vector to be filled. Afterwards it contains uniformly
            distributed random values
        """
        np.copyto(dest, random.uniform(
            self.min_value, self.max_value, len(dest)))
