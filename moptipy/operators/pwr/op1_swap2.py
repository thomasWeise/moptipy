from moptipy.api import operators
from numpy.random import Generator
import numpy as np


class Op1Swap2(operators.Op1):
    """
    This unary search operation for permutations with repetitions swaps two
    elements, i.e., performs exactly one swap.
    It spans a neighborhood of a rather limited size but is easy and fast.
    """

    def op1(self, random: Generator, x: np.ndarray, dest: np.ndarray):
        """
        Create a modified copy of `x` and store it in `dest` by swapping two
        values.
        :param random: the random number generator
        :param x: the existing point in the search space
        :param dest: the array to be shuffled
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
        return "swap2"
