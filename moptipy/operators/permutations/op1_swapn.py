"""
An operator swapping several elements in a permutation.

This operator works similar to
:class:`~moptipy.operators.permutations.op1_swap2.Op1Swap2`, but instead of
swapping two elements (i.e., performing 1 swap), it will perform a random
number of swaps.
"""
from typing import Final, Callable

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op1


# start book
class Op1SwapN(Op1):
    """
    This unary search operation swaps several (different) elements.

    It is similar to `swap2`, but instead may perform a random number
    of swaps. After each swap, it decides with 0.5 probability whether
    or not to perform another swap.
    """

    def op1(self, random: Generator,
            dest: np.ndarray, x: np.ndarray) -> None:
        """
        Copy `x` into `dest` and then swap several different values.

        :param random: the random number generator
        :param dest: the array to receive the modified copy of `x`
        :param x: the existing point in the search space
        """
        np.copyto(dest, x)  # First, we copy `x` to `dest`.
        length: Final[int] = len(dest)  # Get the length of `dest`.
        ri: Final[Callable[[int], int]] = random.integers  # fast call!

        i1: int = ri(length)  # Get the first random index.
        last = first = dest[i1]  # Get the value at the first index.
        continue_after: bool = True
        while continue_after:  # Repeat until we should stop
            continue_after = ri(2) <= 0  # Continue after iteration?
            while True:  # Loop forever until eligible element found.
                i2: int = ri(length)  # Get a new random index.
                current = dest[i2]  # Get the value at the new index.
                if current == last:  # If it is the same as the
                    continue  # previous value, continue.
                if continue_after or (current != first):  # If we want
                    break  # to stop, then it must be != first value.
            dest[i1] = last = current  # Store value for from i2 at i1.
            i1 = i2  # Update i1 to now point to cell of i2.
        dest[i1] = first  # Finally, store first element back at end.
    # end book

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :returns: "swapn", the name of this operator
        :retval "swapn": always
        """
        return "swapn"
