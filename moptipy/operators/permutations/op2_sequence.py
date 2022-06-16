"""
This binary operator merges two permutations by adding elements in sequence.

Assume we have two permutations `x0` and `x1`. For each position `i` in the
destination string `dest`, we randomly select the permutation `x` from which
the value should come (so either `x=x0` or `x=x1`). We then store the first
value not yet marked as done from `x` in `dest[i]`, mark that value as
done both in `x0` and `x1`.
"""
from typing import Final, Callable

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op2
from moptipy.spaces.permutations import Permutations
from moptipy.utils.nputils import DEFAULT_BOOL
from moptipy.utils.types import type_error


# start book
class Op2Sequence(Op2):
    """A binary operator trying to preserve the value sequence."""

    def op2(self, random: Generator, dest: np.ndarray,
            x0: np.ndarray, x1: np.ndarray) -> None:
        """
        Apply a sequence mix from `x0` and `x1` to `dest`.

        :param random: the random number generator
        :param dest: the array to receive the result
        :param x0: the first existing point in the search space
        :param x1: the second existing point in the search space
        """
        # end book
        x0_done: Final[np.ndarray] = self.__x0_done
        x0_done.fill(False)  # all values in x0 are available
        x1_done: Final[np.ndarray] = self.__x1_done
        x1_done.fill(False)  # all values in x1 are available
        ri: Final[Callable[[int], int]] = random.integers
        length: Final[int] = len(x0_done)
        length_minus_1: Final[int] = length - 1
        desti: int = 0  # writing to dest starts at index 0
        x0i: int = 0  # first valid value in x0 is at index 0
        x1i: int = 0  # first valid value in x1 is at index 0

        # start book
        while True:  # repeat until dest is filled, i.e., desti=length
            # randomly chose a source point and pick next operation
            value: int = x0[x0i] if ri(2) <= 0 else x1[x1i]
            dest[desti] = value  # store the value in the destination
            desti = desti + 1  # step destination index

            for x0j in range(x0i, length):  # mark value as done in x0
                if (x0[x0j] == value) and (not x0_done[x0j]):  # find
                    x0_done[x0j] = True  # value is found and not done
                    break  # so we mark it as done and break the loop
            while x0_done[x0i]:  # now we find the next not-yet-done
                x0i = x0i + 1    # value in x0

            if desti >= length_minus_1:
                dest[desti] = x0[x0i]  # store the final missing value
                return  # we are finished, so we return

            for x1j in range(x1i, length):  # mark value as done in x1
                if (x1[x1j] == value) and (not x1_done[x1j]):  # find
                    x1_done[x1j] = True  # value is found and not done
                    break  # so we mark it as done and break the loop
            while x1_done[x1i]:  # now we find the next not-yet-done
                x1i = x1i + 1    # value in x1
    # end book

    def __init__(self, space: Permutations) -> None:
        """
        Initialize the sequence crossover operator.

        :param space: the permutation space
        """
        super().__init__()
        if not isinstance(space, Permutations):
            raise type_error(space, "space", Permutations)
        #: the elements that are done in `x0`
        self.__x0_done: Final[np.ndarray] = np.ndarray(
            (space.dimension, ), DEFAULT_BOOL)
        #: the elements that are done in `x1`
        self.__x1_done: Final[np.ndarray] = np.ndarray(
            (space.dimension, ), DEFAULT_BOOL)

    def __str__(self) -> str:
        """
        Get the name of this binary operator.

        :returns: "sequence", the name of this operator
        :retval "sequence": always
        """
        return "sequence"
