"""
An operator swapping two elements in a permutation or flipping a sign.

This operator is for :mod:`~moptipy.spaces.signed_permutations`. A similar
operator which *only* swaps elements (for :mod:`~moptipy.spaces.permutations`)
is defined in :mod:`~moptipy.operators.permutations.op1_swap2`.
"""
from typing import Final

import numba  # type: ignore
import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op1


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def swap_2_or_flip(random: Generator, dest: np.ndarray,
                   x: np.ndarray) -> None:
    """
    Copy `x` into `dest` and swap two different values or flip a sign.

    :param random: the random number generator
    :param dest: the array to receive the modified copy of `x`
    :param x: the existing point in the search space

    >>> rand = np.random.default_rng(10)
    >>> xx = np.array(range(10), int)
    >>> out = np.empty(len(xx), int)
    >>> swap_2_or_flip(rand, out, xx)
    >>> print(out)
    [0 1 7 3 4 5 6 2 8 9]
    >>> swap_2_or_flip(rand, out, xx)
    >>> print(out)
    [0 1 8 3 4 5 6 7 2 9]
    >>> swap_2_or_flip(rand, out, xx)
    >>> print(out)
    [ 0  1  2  3  4 -5  6  7  8  9]
    >>> swap_2_or_flip(rand, out, xx)
    >>> print(out)
    [0 8 2 3 4 5 6 7 1 9]
    >>> swap_2_or_flip(rand, out, xx)
    >>> print(out)
    [ 0 -1  2  3  4  5  6  7  8  9]
    >>> swap_2_or_flip(rand, out, xx)
    >>> print(out)
    [ 0  1  2  3  4  5 -6  7  8  9]
    """
    dest[:] = x[:]  # First, we copy `x` to `dest`.
    length: Final[int] = len(dest)  # Get the length of `dest`.

    i1: Final[int] = random.integers(0, length)  # first random index.
    v1: Final = dest[i1]  # Get the value at the first index.

    if random.integers(0, 2) == 0:  # With p=0.5, we flip the sign of v1.
        dest[i1] = -v1  # Flip the sign of v1 and store it back at i1.
        return  # Quit.

# Swap two values. Now, normally we would repeat the loop until we find a
# different value. However, if we have signed permutations, it may be
# possible that all values currently are the same. To avoid an endless loop,
# we therefore use a sufficiently large range.
    for _ in range(10 + length):
        i2: int = random.integers(0, length)  # Get the second random index.
        v2 = dest[i2]  # Get the value at the second index.
        if v1 != v2:  # If both values different...
            dest[i2] = v1  # store v1 where v2 was
            dest[i1] = v2  # store v2 where v1 was
            return  # Exit function: we are finished.
    # If we get here, probably all elements are the same, so we just...
    dest[i1] = -v1  # ...flip the sign of v1 and store it back at i1.


class Op1Swap2OrFlip(Op1):
    """
    A search operation that swaps two (different) elements or flips a sign.

    In other words, it performs exactly one swap on a permutation or a sign
    flip. It spans a neighborhood of a rather limited size but is easy
    and fast.
    """

    def __init__(self) -> None:
        """Initialize the object."""
        super().__init__()
        self.op1 = swap_2_or_flip  # type: ignore  # use function directly

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :returns: "swap2orFlip", the name of this operator
        :retval "swap2orFlip": always
        """
        return "swap2orFlip"
