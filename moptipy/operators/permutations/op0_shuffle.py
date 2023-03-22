"""
A nullary operator shuffling a permutation.

This operator first copies the canonical permutation to the destination
string. It then applies the shuffle procedure of the random number
generator, which probably internally applies a Fisher-Yates Shuffle.
The result is a random permutation.

1. Thomas Weise. *Optimization Algorithms.* 2021. Hefei, Anhui, China:
   Institute of Applied Optimization (IAO), School of Artificial Intelligence
   and Big Data, Hefei University. http://thomasweise.github.io/oa/
"""
from typing import Final

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op0
from moptipy.spaces.permutations import Permutations
from moptipy.utils.types import type_error


# start book
class Op0Shuffle(Op0):
    """Shuffle permutations randomly."""

    def __init__(self, space: Permutations):
        """
        Initialize this shuffle operation: use blueprint from space.

        :param space: the search space
        """
        super().__init__()  # -book
        if not isinstance(space, Permutations):  # -book
            raise type_error(space, "space", Permutations)  # -book
        #: the internal blueprint for filling permutations
        self.__blueprint: Final[np.ndarray] = space.blueprint

    def op0(self, random: Generator, dest: np.ndarray) -> None:
        """
        Copy the base string to `dest` and shuffle it randomly.

        :param random: the random number generator
        :param dest: the permutation that should be filled with a random
            sequence of the base permutation.
        """
        np.copyto(dest, self.__blueprint)  # Copy blueprint to dest.
        random.shuffle(dest)  # Shuffle destination array randomly.
    # end book

    def __str__(self) -> str:
        """
        Get the name of this operator.

        :return: "shuffle"
        """
        return "shuffle"
