"""
An operator swapping two elements in a permutation.

This is a unary search operator which first copies the source string `x` to
the destination string `dest`. Then it draws an index `i1` randomly.
It keeps drawing a second random index `i2` until `dest[i1] != dest[i2]`,
i.e., until the elements at the two indices are different. This will always
be true for actual permutations if `i1 != i2`, but for permutations with
repetitions, even if `i1 != i2`, sometimes `dest[i1] == dest[i2]`. Anyway,
as soon as the elements at `i1` and `i2` are different, they will be swapped.

This operator performs one swap. It is similar to :class:`~moptipy.operators.\
permutations.op1_swapn.Op1SwapN`, which performs a random number of swaps.
"""
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
