"""A generalized version of the Order Crossover operator."""
from typing import Final, Callable

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op2
from moptipy.spaces.permutations import Permutations
from moptipy.utils.nputils import DEFAULT_BOOL
from moptipy.utils.types import type_error


# start book
class Op2Order(Op2):
    """A generalized version of the order crossover operator."""

    def op2(self, random: Generator, dest: np.ndarray,
            x0: np.ndarray, x1: np.ndarray) -> None:
        """
        Apply generalized order crossover from `x0` and `x1` to `dest`.

        :param random: the random number generator
        :param dest: the array to receive the result
        :param x0: the first existing point in the search space
        :param x1: the second existing point in the search space
        """
        # end book
        done: Final[np.ndarray] = self.__done
        done.fill(False)  # no value is stored in dest yet
        x1_done: Final[np.ndarray] = self.__x1_done
        x1_done.fill(False)  # all values in x1 are available
        ri: Final[Callable[[int], int]] = random.integers
        length: Final[int] = len(done)  # get length of string
        length_minus_1: Final[int] = length - 1

        # start book
        x1i: int = 0  # first valid value in x1 is at index 0
        for i in range(length):  # enumerate along x0 and dest
            if ri(2) == 0:  # copy value from x0 with probability 0.5
                continue  # skip value with probability 0.5
            value: int = x0[i]  # get value from first permutation
            dest[i] = value  # copy value to dest
            done[i] = True  # mark element of dest as done

            for x1j in range(x1i, length):  # mark value as done in x1
                if (x1[x1j] == value) and (not x1_done[x1j]):  # find
                    x1_done[x1j] = True  # value is found and not done
                    break  # so we mark it as done and break the loop
            if i >= length_minus_1:
                break  # if we are finished, then quit
            while x1_done[x1i]:  # now we find the next not-yet-done
                x1i = x1i + 1    # value in x1

        x1i = 0  # ok, now let's fill the remaining positions in dest
        for i, not_needed in enumerate(done):  # iterate over done
            if not_needed:  # if a position in dest is already filled..
                continue  # ...then skip over it
            while x1_done[x1i]:  # ok, position is not filled, so find
                x1i = x1i + 1  # the first value in x1i not yet used
            dest[i] = x1[x1i]  # and store it in dest
            x1i = x1i + 1   # and move on to the next value
    # end book

    def __init__(self, space: Permutations) -> None:
        """
        Initialize the sequence crossover operator.

        :param space: the permutation space
        """
        super().__init__()
        if not isinstance(space, Permutations):
            raise type_error(space, "space", Permutations)
        #: the elements that are done in `x0` and `dest`
        self.__done: Final[np.ndarray] = np.ndarray(
            (space.dimension, ), DEFAULT_BOOL)
        #: the elements that are done in `x1`
        self.__x1_done: Final[np.ndarray] = np.ndarray(
            (space.dimension, ), DEFAULT_BOOL)

    def __str__(self) -> str:
        """
        Get the name of this binary operator.

        :returns: "order", the name of this operator
        :retval "order": always
        """
        return "order"
