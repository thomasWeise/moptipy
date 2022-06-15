"""A unary operator flipping exactly one bit."""
from typing import Final

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op1


class Op1Flip1(Op1):
    """This unary search operation flips exactly one bit."""

    def op1(self, random: Generator, dest: np.ndarray, x: np.ndarray) -> None:
        """
        Copy `x` into `dest` and flip exactly one bit.

        :param self: the self pointer
        :param random: the random number generator
        :param dest: the array to be shuffled
        :param x: the existing point in the search space
        """
        np.copyto(dest, x)  # copy source to destination
        idx: Final[int] = random.integers(len(x))  # get bit index
        dest[idx] = not dest[idx]  # flip bit

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :return: "flip1"
        """
        return "flip1"
