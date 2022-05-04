"""An operator swapping two elements in a permutation."""
from typing import Final, Callable

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op1


# start book
class Op1Swap2(Op1):
    """
    This unary search operation swaps two (different) elements.

    In other words, it performs exactly one swap on a permutation.
    It spans a neighborhood of a rather limited size but is easy
    and fast.
    """

    def op1(self, random: Generator,
            dest: np.ndarray, x: np.ndarray) -> None:
        """
        Copy `x` into `dest` and swap two different values in `dest`.

        :param random: the random number generator
        :param dest: the array to receive the modified copy of `x`
        :param x: the existing point in the search space
        """
        np.copyto(dest, x)  # First, we copy `x` to `dest`.
        length: Final[int] = len(dest)  # Get the length of `dest`.
        ri: Final[Callable[[int], int]] = random.integers  # fast call!

        i1: Final[int] = ri(length)  # Get the first random index.
        v1: Final = dest[i1]  # Get the value at the first index.
        while True:  # Repeat until we find a different value.
            i2: int = ri(length)  # Get the second random index.
            v2 = dest[i2]  # Get the value at the second index.
            if v1 != v2:  # If both values different...
                dest[i2] = v1  # store v1 where v2 was
                dest[i1] = v2  # store v2 where v1 was
                return  # Exit function: we are finished.
    # end book

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :returns: "swap2", the name of this operator
        :retval "swap2": always
        """
        return "swap2"
