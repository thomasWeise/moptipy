"""A nullary operator filling a vector with uniformly distribute values."""

from typing import Final

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op0
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error


class Op0Uniform(Op0):
    """Fill a vector with uniformly distributed random values."""

    def __init__(self, space: VectorSpace) -> None:
        """
        Initialize the operator.

        :param space: the vector space
        """
        super().__init__()
        if not isinstance(space, VectorSpace):
            raise type_error(space, "space", VectorSpace)
        #: store the space
        self.space: Final[VectorSpace] = space

    def __str__(self) -> str:
        """
        Get the name of this operator.

        :returns: the name of this space
        """
        return "uniform"

    def op0(self, random: Generator, dest: np.ndarray) -> None:
        """
        Fill the string `dest` with random values.

        :param random: the random number generator
        :param dest: the vector to be filled. Afterwards it contains uniformly
            distributed random values
        """
        sp: Final[VectorSpace] = self.space
        np.copyto(dest, random.uniform(
            sp.lower_bound, sp.upper_bound, sp.dimension))

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this operator to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        self.space.log_bounds(logger)
