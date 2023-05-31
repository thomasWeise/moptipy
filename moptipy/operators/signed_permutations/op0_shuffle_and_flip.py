"""
A nullary operator shuffling a signed permutation and flipping signs.

This operator first copies the canonical permutation to the destination
string. It then applies the shuffle procedure of the random number
generator, which probably internally applies a Fisher-Yates Shuffle.
The result is a random permutation. Then, it goes over the permutation
once more and the signs of the elements randomly.
"""
from typing import Final

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op0
from moptipy.spaces.signed_permutations import SignedPermutations
from moptipy.utils.types import type_error


class Op0ShuffleAndFlip(Op0):
    """Shuffle permutations randomly and flip signs randomly."""

    def __init__(self, space: SignedPermutations):
        """
        Initialize this shuffle and flip operation: use blueprint from space.

        :param space: the search space
        """
        super().__init__()
        if not isinstance(space, SignedPermutations):
            raise type_error(space, "space", SignedPermutations)
        #: the internal blueprint for filling permutations
        self.__blueprint: Final[np.ndarray] = space.blueprint

    def op0(self, random: Generator, dest: np.ndarray) -> None:
        """
        Copy the base string to `dest` and shuffle it and flip signs randomly.

        :param random: the random number generator
        :param dest: the signed permutation that should be filled with a
            random sequence of the base permutation with potentially flipped
            signs.
        """
        np.copyto(dest, self.__blueprint)
        random.shuffle(dest)  # Shuffle destination array randomly.
        # integers(0, 2, n) -> n values V in {0, 1}
        # (2 * V) - 1 -> a value in {-1, 1}
        # dest[1] * V -> a value which is either {-dest[i], dest[i]}
        dest *= ((2 * random.integers(low=0, high=2, size=len(dest))) - 1)

    def __str__(self) -> str:
        """
        Get the name of this shuffle-and-flip operator.

        :return: "shuffleAndFlip"
        """
        return "shuffleAndFlip"
