"""A generalized version of the Order Crossover operator."""
from typing import Final, Callable

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op2
from moptipy.spaces.permutations import Permutations
from moptipy.utils.nputils import DEFAULT_BOOL, DEFAULT_INT
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
        indices: Final[np.ndarray] = self.__indices
        x1_done: Final[np.ndarray] = self.__x1_done
        x1_done.fill(False)  # all values in x1 are available
        ri: Final[Callable[[int], int]] = random.integers
        rbin: Final[Callable[[int, float], int]] = random.binomial
        length: Final[int] = len(indices)  # get length of string
        copy_from_x0: int  # the end index of copying from x0
        value: int  # the current value to be written to dest

        # start book
        while True:  # sample the number of values to copy from x0
            copy_from_x0 = rbin(length, 0.5)  # p=0.5 for each value
            if 1 < copy_from_x0 < (length - 1):  # ensure difference
                break                      # from each parent
        copy_from_x0 = length - copy_from_x0  # compute end index

        i: int = length  # this is the index we iterate over
        mode: bool = True  # mode: True = copy from x0, False = from x1
        x1i: int = 0  # the index of the next unused value from x1
        while True:  # loop until we are finished
            index_i: int = ri(i)  # pick a random index-index
            index: int = indices[index_i]  # load the actual index
            i = i - 1  # reduce the number of values
            indices[i], indices[index_i] = index, indices[i]  # swap

            if mode:  # copy from x0 to dest
                dest[index] = value = x0[index]  # get value
                for x1j in range(x1i, length):  # mark as used
                    if (x1[x1j] == value) and (not x1_done[x1j]):
                        x1_done[x1j] = True  # mark value as used
                        break  # exit inner loop
                if copy_from_x0 == i:  # are we done with copying?
                    mode = False  # set mode to load from x1
                    x1i = 0  # reset x1 index
            else:  # go to next iteration
                dest[index] = x1[x1i]  # and store it in dest
                if i == 0:  # check if we are done
                    return  # ok, we are finished
                x1i = x1i + 1  # and move on to the next value
            while x1_done[x1i]:  # step x1i to next unused value
                x1i = x1i + 1  # increment
    # end book

    def __init__(self, space: Permutations) -> None:
        """
        Initialize the sequence crossover operator.

        :param space: the permutation space
        """
        super().__init__()
        if not isinstance(space, Permutations):
            raise type_error(space, "space", Permutations)
        if space.dimension < 4:
            raise ValueError(
                f"dimension must be > 3, but got {space.dimension}.")
        #: the valid indices
        self.__indices: Final[np.ndarray] = np.array(
            range(space.dimension), DEFAULT_INT)
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
