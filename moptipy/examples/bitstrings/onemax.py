"""
An objective function counting the number of ones in a bit string.

1. Heinz Mühlenbein. How Genetic Algorithms Really Work: Mutation and
   Hillclimbing. In Reinhard Männer and Bernard Manderick, editors,
   *Proceedings of Parallel Problem Solving from Nature 2 (PPSN-II),*
   September 28-30, 1992, Brussels, Belgium, pages 15-26. Elsevier.
   https://www.researchgate.net/publication/220702092
2. Stefan Droste, Thomas Jansen, and Ingo Wegener. Upper and Lower Bounds for
   Randomized Search Heuristics in Black-Box Optimization. *Theory of
   Computing Systems.* 39(4):525-544. July 2006.
   doi: https://doi.org/10.1007/s00224-004-1177-z
"""

import numba  # type: ignore
import numpy as np

from moptipy.examples.bitstrings.bitstring_problem import BitStringProblem


@numba.njit(nogil=True, cache=True)
def onemax(x: np.ndarray) -> int:
    """
    Get the length of a string minus the number of ones in it.

    :param x: the np array
    :return: the length of the string minus the number of ones, i.e., the
        number of zeros

    >>> print(onemax(np.array([True, True, False, False, False])))
    3
    >>> print(onemax(np.array([True, False, True, False, False])))
    3
    >>> print(onemax(np.array([False, True,  True, False, False])))
    3
    >>> print(onemax(np.array([True, True, True, True, True])))
    0
    >>> print(onemax(np.array([False, True, True, True, True])))
    1
    >>> print(onemax(np.array([False, False, False, False, False])))
    5
    """
    return int(len(x) - x.sum())


class OneMax(BitStringProblem):
    """Maximize the number of ones in a bit string."""

    def __init__(self, n: int) -> None:  # +book
        """
        Initialize the onemax objective function.

        :param n: the dimension of the problem

        >>> print(OneMax(2).n)
        2
        >>> print(OneMax(4).evaluate(np.array([True, True, False, True])))
        1
        """
        super().__init__(n)
        self.evaluate = onemax  # type: ignore

    def __str__(self) -> str:
        """
        Get the name of the onemax objective function.

        :return: `onemax_` + length of string

        >>> print(OneMax(13))
        onemax_13
        """
        return f"onemax_{self.n}"
