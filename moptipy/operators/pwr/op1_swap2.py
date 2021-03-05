"""An operator swapping two elements in a permutation with repetitions."""
import numpy as np
from numpy.random import Generator

from moptipy.api import operators


class Op1Swap2(operators.Op1):
    """
    This unary search operation swaps two different elements.

    In other words, it performs exactly one swap on a permutation with
    repetitions. It spans a neighborhood of a rather limited size but is
    easy and fast.
    """

    def op1(self, random: Generator, x: np.ndarray, dest: np.ndarray) -> None:
        """
        Create a copy `x` into `dest` and swap two different values in `dest`.

        :param Generator random: the random number generator
        :param np.ndarray x: the existing point in the search space
        :param np.ndarray dest: the array to be shuffled
        """
        np.copyto(dest, x)
        length = len(dest)
        i1 = random.integers(length)
        v1 = dest[i1]
        while True:
            i2 = random.integers(length)
            v2 = dest[i2]
            if v1 != v2:
                dest[i2] = v1
                dest[i1] = v2
                return

    def get_name(self) -> str:
        """
        Get the name of this unary operator.

        :return: "swap2"
        :rtype: str
        """
        return "swap2"
