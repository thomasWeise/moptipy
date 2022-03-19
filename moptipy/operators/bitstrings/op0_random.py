"""A nullary operator filling a bit string with random values."""

import numpy as np
from numpy.random import Generator

from moptipy.api import operators


class Op0Random(operators.Op0):
    """Fill a bit string with random values."""

    def op0(self, random: Generator, dest: np.ndarray) -> None:
        """
        Fill the string `dest` with radom values.

        :param Generator random: the random number generator
        :param np.ndarray dest: the bit string to fill. Afterwards, its
            contents will be random.
        """
        np.copyto(dest, random.integers(0, 2, dest.shape, dest.dtype))

    def __str__(self) -> str:
        """
        Get the name of this operator.

        :return: "randomize"
        :rtype: str
        """
        return "randomize"