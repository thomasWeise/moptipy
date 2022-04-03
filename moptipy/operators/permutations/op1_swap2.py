"""An operator swapping two elements in a permutation."""
from typing import Final, Callable

import numpy as np
from numpy.random import Generator

from moptipy.api import operators


class Op1Swap2(operators.Op1):
    """
    This unary search operation swaps two different elements.

    In other words, it performs exactly one swap on a permutation.
    It spans a neighborhood of a rather limited size but is easy
    and fast.
    """

    def op1(self, random: Generator, dest: np.ndarray, x: np.ndarray) -> None:
        """
        Create a copy `x` into `dest` and swap two different values in `dest`.

        :param random: the random number generator
        :param dest: the array to be shuffled
        :param x: the existing point in the search space
        """
        np.copyto(dest, x)  # first copy source to dest
        length: Final[int] = len(dest)  # get the length
        ri: Final[Callable[[int], int]] = random.integers  # fast call!

        i1: Final[int] = ri(length)  # get first index
        v1: Final = dest[i1]  # get first value
        while True:  # repeat until we find different value
            i2: int = ri(length)  # get second index (fast call!)
            v2 = dest[i2]  # get second value
            if v1 != v2:  # are both values different?
                dest[i2] = v1  # store v1 where v2 was
                dest[i1] = v2  # store v2 where v1 was
                return

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :return: "swap2"
        """
        return "swap2"
