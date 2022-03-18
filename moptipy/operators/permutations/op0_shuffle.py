"""A nullary operator shuffling a permutation."""
from typing import Final

import numpy as np
from numpy.random import Generator

from moptipy.api import operators
from moptipy.spaces.permutations import Permutations


# start book
class Op0Shuffle(operators.Op0):
    """Shuffle permutations randomly."""

    def __init__(self, space: Permutations):
        """
        Initialize this shuffle operation: use blueprint from space.

        :param Permutations space: the search space
        """
        if not space:  # -book
            raise ValueError("space must not be None.")  # -book
        if not isinstance(space, Permutations):  # -book
            raise TypeError(  # -book
                "space must be PermutationsWithRepetitions,"  # -book
                f" but is {type(space)}.")  # -book
        #: the internal blueprint for filling permutations
        self.__blueprint: Final[np.ndarray] = space.blueprint

    def op0(self, random: Generator, dest: np.ndarray) -> None:
        """
        Copy the base string to `dest` and shuffle it randomly.

        :param Generator random: the random number generator
        :param np.ndarray dest: the permutation that should be shuffled.
            Afterwards, the order of its elements if random.
        """
        np.copyto(dest, self.__blueprint)
        random.shuffle(dest)
    # end book

    def __str__(self) -> str:
        """
        Get the name of this operator.

        :return: "shuffle"
        :rtype: str
        """
        return "shuffle"
