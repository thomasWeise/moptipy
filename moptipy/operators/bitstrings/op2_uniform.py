"""A binary operator copying each bit from either source string."""
from typing import Callable, Final

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op2


class Op2Uniform(Op2):
    """
    This binary search operation copies each bit from either source.

    For each index `i` in the destination array `dest`, uniform
    crossover copies the value from the first source string `x0`with
    probability 0.5 and otherwise the value from the second source
    string `x1`. All bits that have the same value in `x0` and `x1`
    will retain this value in `dest`, all bits where `x0` and `x1`
    differ will effectively be randomized (be `0` with probability 0.5
    and `1` with probability 0.5).
    """

    def op2(self, random: Generator, dest: np.ndarray, x0: np.ndarray,
            x1: np.ndarray) -> None:
        """
        Store result of uniform crossover of `x0` and `x1` in `dest`.

        :param self: the self pointer
        :param random: the random number generator
        :param dest: the array to receive the result
        :param x1: the first existing point in the search space
        :param x2: the second existing point in the search space
        """
        np.copyto(dest, x0)  # copy first source to destination
        ri: Final[Callable[[int], int]] = random.integers
        for i, v in enumerate(x1):  # iterate over second source
            if ri(2) <= 0:  # probability 0.5 = 50% chance to...
                dest[i] = v  # ...copy value from second source

    def __str__(self) -> str:
        """
        Get the name of this binary operator.

        :return: "uniform"
        """
        return "uniform"
