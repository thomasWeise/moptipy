"""A binary operator copying each bit from either source string."""

import numba  # type: ignore
import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op2


@numba.njit(nogil=True, cache=True)
def __op2_uniform(r: np.ndarray, dest: np.ndarray, x0: np.ndarray,
                  x1: np.ndarray) -> None:
    """
    Perform the actual work of the uniform crossover.

    :param r: the array with random numbers in 0..1
    :param dest: the destination array
    :param x0: the first source array
    :param x1: the second source array
    """
    for i, v in enumerate(r):  # iterate over random numbers
        # copy from x0 with p=0.5 and from x1 with p=0.5
        dest[i] = x1[i] if v == 0 else x0[i]


def _op2_uniform(random: Generator, dest: np.ndarray, x0: np.ndarray,
                 x1: np.ndarray) -> None:
    """
    Perform the uniform operator as plain old function.

    :param random: the random number generator
    :param dest: the array to receive the result
    :param x0: the first source array
    :param x1: the second source array
    """
    __op2_uniform(random.integers(low=2, high=None, size=len(dest)),
                  dest, x0, x1)


class Op2Uniform(Op2):
    """
    This binary search operation copies each bit from either source.

    For each index `i` in the destination array `dest`, uniform
    crossover copies the value from the first source string `x0`with
    probability 0.5 and otherwise the value from the second source
    string `x1`. All bits that have the same value in `x0` and `x1`
    will retain this value in `dest`, all bits where `x0` and `x1`
    differ will effectively be randomized (be `0` with probability 0.5
    and `1` with probability 0.5).
    """

    def __init__(self):
        """Initialize the uniform crossover operator."""
        super().__init__()
        self.op2 = _op2_uniform  # type: ignore

    def __str__(self) -> str:
        """
        Get the name of this binary operator.

        :return: "uniform"
        """
        return "uniform"
