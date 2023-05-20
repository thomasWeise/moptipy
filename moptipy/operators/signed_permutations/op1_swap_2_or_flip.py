"""
An operator swapping two elements in a permutation or flipping a sign.

This operator is for :mod:`~moptipy.spaces.signed_permutations`. A similar
operator which *only* swaps elements (for :mod:`~moptipy.spaces.permutations`)
is defined in :mod:`~moptipy.operators.permutations.op1_swap2`.
"""
from typing import Callable, Final

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op1


# start book
class Op1Swap2OrFlip(Op1):
    """
    A unary search operation that swaps two (different) elements or a sign.

    In other words, it performs exactly one swap on a permutation or a sign
    flip. It spans a neighborhood of a rather limited size but is easy
    and fast.
    """

    def op1(self, random: Generator,
            dest: np.ndarray, x: np.ndarray) -> None:
        """
        Copy `x` into `dest` and swap two different values or flip a sign.

        :param random: the random number generator
        :param dest: the array to receive the modified copy of `x`
        :param x: the existing point in the search space
        """
        np.copyto(dest, x)  # First, we copy `x` to `dest`.
        length: Final[int] = len(dest)  # Get the length of `dest`.
        ri: Final[Callable[[int], int]] = random.integers  # fast call!

        if ri(2) == 0:  # With probability 0.5, we flip a sign.
            dest[ri(length)] *= -1  # x * -1 == -x
            return  # Quit.

        i1: Final[int] = ri(length)  # Get the first random index.
        v1: Final = dest[i1]  # Get the value at the first index.
        while True:  # Repeat until we find a different value.
            i2: int = ri(length)  # Get the second random index.
            v2 = dest[i2]  # Get the value at the second index.
            if v1 != v2:  # If both values different...
                dest[i2] = v1  # store v1 where v2 was
                dest[i1] = v2  # store v2 where v1 was
                return  # Exit function: we are finished.

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :returns: "swap2orFlip", the name of this operator
        :retval "swap2orFlip": always
        """
        return "swap2orFlip"
