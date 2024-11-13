"""
An objective function counting the number of ones in a bit string.

This function tries to maximize the number of `True` bits in a bit string.
Therefore, it returns the lowest possible value 0 if all bits are `True`.
This is the global optimum.
It returns the highest possible value `n` if all bits are `False`. This is
the worst possible value.

1. Heinz Mühlenbein. How Genetic Algorithms Really Work: Mutation and
   Hillclimbing. In Reinhard Männer and Bernard Manderick, editors,
   *Proceedings of Parallel Problem Solving from Nature 2 (PPSN-II),*
   September 28-30, 1992, Brussels, Belgium, pages 15-26. Elsevier.
   https://www.researchgate.net/publication/220702092
2. Stefan Droste, Thomas Jansen, and Ingo Wegener. Upper and Lower Bounds for
   Randomized Search Heuristics in Black-Box Optimization. *Theory of
   Computing Systems.* 39(4):525-544. July 2006.
   doi: https://doi.org/10.1007/s00224-004-1177-z
3. Thomas Weise, Zhize Wu, Xinlu Li, and Yan Chen. Frequency Fitness
   Assignment: Making Optimization Algorithms Invariant under Bijective
   Transformations of the Objective Function Value. *IEEE Transactions on
   Evolutionary Computation* 25(2):307-319. April 2021. Preprint available at
   arXiv:2001.01416v5 [cs.NE] 15 Oct 2020.
   https://dx.doi.org/10.1109/TEVC.2020.3032090
4. v
"""

from typing import Callable, Iterator, cast

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

    >>> onemax(np.array([True, True, False, False, False]))
    3
    >>> onemax(np.array([True, False, True, False, False]))
    3
    >>> onemax(np.array([False, True,  True, False, False]))
    3
    >>> onemax(np.array([True, True, True, True, True]))
    0
    >>> onemax(np.array([False, True, True, True, True]))
    1
    >>> onemax(np.array([False, False, False, False, False]))
    5

    # n = 1 and 0 true bits
    >>> onemax(np.array([0]))
    1

    # n = 1 and 1 true bit
    >>> onemax(np.array([1]))
    0

    # n = 2 and 0 true bits
    >>> onemax(np.array([0, 0]))
    2

    # n = 2 and 1 true bit
    >>> onemax(np.array([0, 1]))
    1

    # n = 2 and 1 true bit
    >>> onemax(np.array([1, 0]))
    1

    # n = 2 and 1 true bit
    >>> onemax(np.array([0, 1]))
    1

    # n = 2 and 2 true bits
    >>> onemax(np.array([1, 1]))
    0

    # n = 3 and 0 true bits
    >>> onemax(np.array([0, 0, 0]))
    3

    # n = 3 and 1 true bit
    >>> onemax(np.array([1, 0, 0]))
    2

    # n = 3 and 1 true bit
    >>> onemax(np.array([0, 1, 0]))
    2

    # n = 3 and 1 true bit
    >>> onemax(np.array([0, 0, 1]))
    2

    # n = 3 and 2 true bits
    >>> onemax(np.array([1, 1, 0]))
    1

    # n = 3 and 2 true bits
    >>> onemax(np.array([0, 1, 1]))
    1

    # n = 3 and 2 true bits
    >>> onemax(np.array([1, 1, 0]))
    1

    # n = 3 and 3 true bits
    >>> onemax(np.array([1, 1, 1]))
    0

    # n = 4 and 0 true bits
    >>> onemax(np.array([0, 0, 0, 0]))
    4

    # n = 4 and 1 true bit
    >>> onemax(np.array([1, 0, 0, 0]))
    3

    # n = 4 and 1 true bit
    >>> onemax(np.array([0, 0, 0, 1]))
    3

    # n = 4 and 1 true bit
    >>> onemax(np.array([0, 0, 1, 0]))
    3

    # n = 4 and 2 true bits
    >>> onemax(np.array([0, 0, 1, 1]))
    2

    # n = 4 and 2 true bits
    >>> onemax(np.array([0, 1, 0, 1]))
    2

    # n = 4 and 2 true bits
    >>> onemax(np.array([0, 1, 0, 1]))
    2

    # n = 4 and 3 true bits
    >>> onemax(np.array([0, 1, 1, 1]))
    1

    # n = 4 and 3 true bits
    >>> onemax(np.array([1, 1, 1, 0]))
    1

    # n = 4 and 3 true bits
    >>> onemax(np.array([1, 0, 1, 1]))
    1

    # n = 4 and 4 true bits
    >>> onemax(np.array([1, 1, 1, 1]))
    0

    # n = 5 and 0 true bits
    >>> onemax(np.array([0, 0, 0, 0, 0]))
    5

    # n = 5 and 1 true bit
    >>> onemax(np.array([0, 0, 0, 0, 1]))
    4

    # n = 5 and 1 true bit
    >>> onemax(np.array([0, 1, 0, 0, 0]))
    4

    # n = 5 and 1 true bit
    >>> onemax(np.array([1, 0, 0, 0, 0]))
    4

    # n = 5 and 2 true bits
    >>> onemax(np.array([0, 1, 0, 0, 1]))
    3

    # n = 5 and 2 true bits
    >>> onemax(np.array([0, 1, 1, 0, 0]))
    3

    # n = 5 and 2 true bits
    >>> onemax(np.array([0, 0, 0, 1, 1]))
    3

    # n = 5 and 3 true bits
    >>> onemax(np.array([1, 0, 1, 1, 0]))
    2

    # n = 5 and 3 true bits
    >>> onemax(np.array([1, 1, 0, 1, 0]))
    2

    # n = 5 and 3 true bits
    >>> onemax(np.array([0, 1, 1, 1, 0]))
    2

    # n = 5 and 4 true bits
    >>> onemax(np.array([1, 0, 1, 1, 1]))
    1

    # n = 5 and 4 true bits
    >>> onemax(np.array([1, 1, 0, 1, 1]))
    1

    # n = 5 and 4 true bits
    >>> onemax(np.array([1, 0, 1, 1, 1]))
    1

    # n = 5 and 5 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1]))
    0

    # n = 6 and 0 true bits
    >>> onemax(np.array([0, 0, 0, 0, 0, 0]))
    6

    # n = 6 and 1 true bit
    >>> onemax(np.array([0, 0, 0, 0, 1, 0]))
    5

    # n = 6 and 1 true bit
    >>> onemax(np.array([1, 0, 0, 0, 0, 0]))
    5

    # n = 6 and 1 true bit
    >>> onemax(np.array([0, 0, 0, 0, 1, 0]))
    5

    # n = 6 and 2 true bits
    >>> onemax(np.array([0, 1, 0, 1, 0, 0]))
    4

    # n = 6 and 2 true bits
    >>> onemax(np.array([0, 1, 0, 0, 1, 0]))
    4

    # n = 6 and 2 true bits
    >>> onemax(np.array([0, 0, 0, 0, 1, 1]))
    4

    # n = 6 and 3 true bits
    >>> onemax(np.array([0, 0, 1, 0, 1, 1]))
    3

    # n = 6 and 3 true bits
    >>> onemax(np.array([1, 0, 0, 1, 0, 1]))
    3

    # n = 6 and 3 true bits
    >>> onemax(np.array([0, 1, 1, 0, 0, 1]))
    3

    # n = 6 and 4 true bits
    >>> onemax(np.array([1, 1, 1, 0, 1, 0]))
    2

    # n = 6 and 4 true bits
    >>> onemax(np.array([1, 1, 0, 0, 1, 1]))
    2

    # n = 6 and 4 true bits
    >>> onemax(np.array([1, 1, 1, 0, 1, 0]))
    2

    # n = 6 and 5 true bits
    >>> onemax(np.array([1, 1, 0, 1, 1, 1]))
    1

    # n = 6 and 5 true bits
    >>> onemax(np.array([1, 1, 1, 1, 0, 1]))
    1

    # n = 6 and 5 true bits
    >>> onemax(np.array([1, 1, 1, 1, 0, 1]))
    1

    # n = 6 and 6 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 1]))
    0

    # n = 7 and 0 true bits
    >>> onemax(np.array([0, 0, 0, 0, 0, 0, 0]))
    7

    # n = 7 and 1 true bit
    >>> onemax(np.array([0, 1, 0, 0, 0, 0, 0]))
    6

    # n = 7 and 1 true bit
    >>> onemax(np.array([0, 0, 0, 0, 1, 0, 0]))
    6

    # n = 7 and 1 true bit
    >>> onemax(np.array([0, 1, 0, 0, 0, 0, 0]))
    6

    # n = 7 and 2 true bits
    >>> onemax(np.array([1, 0, 0, 0, 1, 0, 0]))
    5

    # n = 7 and 2 true bits
    >>> onemax(np.array([0, 1, 0, 1, 0, 0, 0]))
    5

    # n = 7 and 2 true bits
    >>> onemax(np.array([1, 0, 0, 0, 0, 0, 1]))
    5

    # n = 7 and 3 true bits
    >>> onemax(np.array([1, 0, 1, 1, 0, 0, 0]))
    4

    # n = 7 and 3 true bits
    >>> onemax(np.array([0, 0, 1, 1, 0, 0, 1]))
    4

    # n = 7 and 3 true bits
    >>> onemax(np.array([0, 0, 1, 1, 0, 0, 1]))
    4

    # n = 7 and 4 true bits
    >>> onemax(np.array([0, 1, 0, 1, 1, 1, 0]))
    3

    # n = 7 and 4 true bits
    >>> onemax(np.array([1, 1, 1, 0, 1, 0, 0]))
    3

    # n = 7 and 4 true bits
    >>> onemax(np.array([0, 1, 1, 1, 0, 1, 0]))
    3

    # n = 7 and 5 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 0, 0]))
    2

    # n = 7 and 5 true bits
    >>> onemax(np.array([1, 1, 1, 1, 0, 1, 0]))
    2

    # n = 7 and 5 true bits
    >>> onemax(np.array([0, 1, 1, 1, 1, 1, 0]))
    2

    # n = 7 and 6 true bits
    >>> onemax(np.array([1, 1, 1, 1, 0, 1, 1]))
    1

    # n = 7 and 6 true bits
    >>> onemax(np.array([1, 1, 1, 0, 1, 1, 1]))
    1

    # n = 7 and 6 true bits
    >>> onemax(np.array([0, 1, 1, 1, 1, 1, 1]))
    1

    # n = 7 and 7 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 8 and 0 true bits
    >>> onemax(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    8

    # n = 8 and 1 true bit
    >>> onemax(np.array([0, 0, 0, 0, 0, 1, 0, 0]))
    7

    # n = 8 and 1 true bit
    >>> onemax(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    7

    # n = 8 and 1 true bit
    >>> onemax(np.array([0, 0, 0, 0, 0, 0, 0, 1]))
    7

    # n = 8 and 2 true bits
    >>> onemax(np.array([0, 0, 0, 0, 0, 0, 1, 1]))
    6

    # n = 8 and 2 true bits
    >>> onemax(np.array([0, 0, 1, 0, 0, 1, 0, 0]))
    6

    # n = 8 and 2 true bits
    >>> onemax(np.array([0, 0, 0, 0, 0, 1, 1, 0]))
    6

    # n = 8 and 3 true bits
    >>> onemax(np.array([0, 1, 0, 0, 1, 1, 0, 0]))
    5

    # n = 8 and 3 true bits
    >>> onemax(np.array([0, 1, 0, 1, 1, 0, 0, 0]))
    5

    # n = 8 and 3 true bits
    >>> onemax(np.array([0, 0, 0, 0, 1, 1, 1, 0]))
    5

    # n = 8 and 4 true bits
    >>> onemax(np.array([1, 0, 1, 1, 0, 0, 1, 0]))
    4

    # n = 8 and 4 true bits
    >>> onemax(np.array([1, 1, 0, 0, 1, 1, 0, 0]))
    4

    # n = 8 and 4 true bits
    >>> onemax(np.array([1, 0, 0, 0, 1, 1, 0, 1]))
    4

    # n = 8 and 5 true bits
    >>> onemax(np.array([0, 1, 1, 0, 0, 1, 1, 1]))
    3

    # n = 8 and 5 true bits
    >>> onemax(np.array([1, 1, 0, 0, 1, 0, 1, 1]))
    3

    # n = 8 and 5 true bits
    >>> onemax(np.array([1, 1, 0, 0, 1, 1, 0, 1]))
    3

    # n = 8 and 6 true bits
    >>> onemax(np.array([1, 1, 0, 1, 1, 1, 1, 0]))
    2

    # n = 8 and 6 true bits
    >>> onemax(np.array([1, 1, 0, 1, 1, 0, 1, 1]))
    2

    # n = 8 and 6 true bits
    >>> onemax(np.array([0, 0, 1, 1, 1, 1, 1, 1]))
    2

    # n = 8 and 7 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 1, 0, 1]))
    1

    # n = 8 and 7 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 1, 1, 0]))
    1

    # n = 8 and 7 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 0, 1, 1]))
    1

    # n = 8 and 8 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 9 and 0 true bits
    >>> onemax(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
    9

    # n = 9 and 1 true bit
    >>> onemax(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]))
    8

    # n = 9 and 1 true bit
    >>> onemax(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]))
    8

    # n = 9 and 1 true bit
    >>> onemax(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]))
    8

    # n = 9 and 2 true bits
    >>> onemax(np.array([1, 0, 0, 0, 0, 0, 0, 1, 0]))
    7

    # n = 9 and 2 true bits
    >>> onemax(np.array([0, 1, 0, 0, 0, 0, 1, 0, 0]))
    7

    # n = 9 and 2 true bits
    >>> onemax(np.array([0, 0, 1, 0, 1, 0, 0, 0, 0]))
    7

    # n = 9 and 3 true bits
    >>> onemax(np.array([0, 0, 1, 0, 0, 1, 0, 0, 1]))
    6

    # n = 9 and 3 true bits
    >>> onemax(np.array([1, 0, 0, 0, 0, 0, 0, 1, 1]))
    6

    # n = 9 and 3 true bits
    >>> onemax(np.array([0, 0, 1, 0, 1, 0, 0, 1, 0]))
    6

    # n = 9 and 4 true bits
    >>> onemax(np.array([0, 0, 0, 0, 1, 1, 0, 1, 1]))
    5

    # n = 9 and 4 true bits
    >>> onemax(np.array([0, 0, 0, 1, 0, 0, 1, 1, 1]))
    5

    # n = 9 and 4 true bits
    >>> onemax(np.array([0, 0, 1, 1, 1, 0, 0, 0, 1]))
    5

    # n = 9 and 5 true bits
    >>> onemax(np.array([0, 0, 1, 1, 0, 1, 0, 1, 1]))
    4

    # n = 9 and 5 true bits
    >>> onemax(np.array([0, 0, 1, 1, 0, 1, 0, 1, 1]))
    4

    # n = 9 and 5 true bits
    >>> onemax(np.array([1, 1, 0, 0, 1, 0, 0, 1, 1]))
    4

    # n = 9 and 6 true bits
    >>> onemax(np.array([0, 0, 1, 1, 1, 1, 1, 0, 1]))
    3

    # n = 9 and 6 true bits
    >>> onemax(np.array([1, 1, 0, 0, 1, 0, 1, 1, 1]))
    3

    # n = 9 and 6 true bits
    >>> onemax(np.array([0, 1, 1, 1, 0, 1, 1, 0, 1]))
    3

    # n = 9 and 7 true bits
    >>> onemax(np.array([0, 1, 1, 0, 1, 1, 1, 1, 1]))
    2

    # n = 9 and 7 true bits
    >>> onemax(np.array([1, 1, 0, 1, 1, 1, 1, 0, 1]))
    2

    # n = 9 and 7 true bits
    >>> onemax(np.array([1, 1, 1, 0, 1, 1, 0, 1, 1]))
    2

    # n = 9 and 8 true bits
    >>> onemax(np.array([1, 1, 1, 0, 1, 1, 1, 1, 1]))
    1

    # n = 9 and 8 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 0, 1, 1, 1]))
    1

    # n = 9 and 8 true bits
    >>> onemax(np.array([1, 1, 1, 0, 1, 1, 1, 1, 1]))
    1

    # n = 9 and 9 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 10 and 0 true bits
    >>> onemax(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    10

    # n = 10 and 1 true bit
    >>> onemax(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
    9

    # n = 10 and 1 true bit
    >>> onemax(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    9

    # n = 10 and 1 true bit
    >>> onemax(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    9

    # n = 10 and 2 true bits
    >>> onemax(np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0]))
    8

    # n = 10 and 2 true bits
    >>> onemax(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    8

    # n = 10 and 2 true bits
    >>> onemax(np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
    8

    # n = 10 and 3 true bits
    >>> onemax(np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0]))
    7

    # n = 10 and 3 true bits
    >>> onemax(np.array([1, 0, 0, 0, 0, 0, 1, 0, 1, 0]))
    7

    # n = 10 and 3 true bits
    >>> onemax(np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0]))
    7

    # n = 10 and 4 true bits
    >>> onemax(np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 0]))
    6

    # n = 10 and 4 true bits
    >>> onemax(np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1]))
    6

    # n = 10 and 4 true bits
    >>> onemax(np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0]))
    6

    # n = 10 and 5 true bits
    >>> onemax(np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0]))
    5

    # n = 10 and 5 true bits
    >>> onemax(np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 1]))
    5

    # n = 10 and 5 true bits
    >>> onemax(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
    5

    # n = 10 and 6 true bits
    >>> onemax(np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0]))
    4

    # n = 10 and 6 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 0, 1, 0, 0, 0]))
    4

    # n = 10 and 6 true bits
    >>> onemax(np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1]))
    4

    # n = 10 and 7 true bits
    >>> onemax(np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 1]))
    3

    # n = 10 and 7 true bits
    >>> onemax(np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 0]))
    3

    # n = 10 and 7 true bits
    >>> onemax(np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 1]))
    3

    # n = 10 and 8 true bits
    >>> onemax(np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 1]))
    2

    # n = 10 and 8 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 0]))
    2

    # n = 10 and 8 true bits
    >>> onemax(np.array([1, 1, 1, 1, 0, 1, 0, 1, 1, 1]))
    2

    # n = 10 and 9 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1]))
    1

    # n = 10 and 9 true bits
    >>> onemax(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    1

    # n = 10 and 9 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1]))
    1

    # n = 10 and 10 true bits
    >>> onemax(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    0
    """
    return len(x) - int(x.sum())


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

        >>> OneMax(13)
        onemax_13
        """
        return f"onemax_{self.n}"

    @classmethod
    def default_instances(
            cls: type, scale_min: int = 2, scale_max: int = 8192) \
            -> Iterator[Callable[[], "OneMax"]]:
        """
        Get the 202 default instances of the :class:`OneMax` problem.

        :param scale_min: the minimum permitted scale, by default `2`
        :param scale_max: the maximum permitted scale, by default `8192`
        :returns: a sequence of default :class:`OneMax` instances

        >>> len(list(OneMax.default_instances()))
        202

        >>> [x() for x in OneMax.default_instances()]
        [onemax_2, onemax_3, onemax_4, onemax_5, onemax_6, onemax_7, \
onemax_8, onemax_9, onemax_10, onemax_11, onemax_12, onemax_13, onemax_14, \
onemax_15, onemax_16, onemax_17, onemax_18, onemax_19, onemax_20, onemax_21, \
onemax_22, onemax_23, onemax_24, onemax_25, onemax_26, onemax_27, onemax_28, \
onemax_29, onemax_30, onemax_31, onemax_32, onemax_33, onemax_36, onemax_40, \
onemax_41, onemax_42, onemax_44, onemax_48, onemax_49, onemax_50, onemax_55, \
onemax_59, onemax_60, onemax_64, onemax_66, onemax_70, onemax_77, onemax_79, \
onemax_80, onemax_81, onemax_85, onemax_88, onemax_90, onemax_96, onemax_99, \
onemax_100, onemax_107, onemax_111, onemax_121, onemax_125, onemax_128, \
onemax_144, onemax_149, onemax_169, onemax_170, onemax_192, onemax_196, \
onemax_199, onemax_200, onemax_222, onemax_225, onemax_243, onemax_256, \
onemax_269, onemax_289, onemax_300, onemax_324, onemax_333, onemax_341, \
onemax_343, onemax_359, onemax_361, onemax_384, onemax_400, onemax_441, \
onemax_444, onemax_479, onemax_484, onemax_500, onemax_512, onemax_529, \
onemax_555, onemax_576, onemax_600, onemax_625, onemax_641, onemax_666, \
onemax_676, onemax_682, onemax_700, onemax_729, onemax_768, onemax_777, \
onemax_784, onemax_800, onemax_841, onemax_857, onemax_888, onemax_900, \
onemax_961, onemax_999, onemax_1000, onemax_1024, onemax_1089, onemax_1111, \
onemax_1151, onemax_1156, onemax_1225, onemax_1296, onemax_1365, \
onemax_1369, onemax_1444, onemax_1521, onemax_1536, onemax_1543, \
onemax_1600, onemax_1681, onemax_1764, onemax_1849, onemax_1936, \
onemax_2000, onemax_2025, onemax_2048, onemax_2063, onemax_2116, \
onemax_2187, onemax_2209, onemax_2222, onemax_2304, onemax_2401, \
onemax_2500, onemax_2601, onemax_2704, onemax_2730, onemax_2753, \
onemax_2809, onemax_2916, onemax_3000, onemax_3025, onemax_3072, \
onemax_3125, onemax_3136, onemax_3249, onemax_3333, onemax_3364, \
onemax_3481, onemax_3600, onemax_3671, onemax_3721, onemax_3844, \
onemax_3969, onemax_4000, onemax_4096, onemax_4225, onemax_4356, \
onemax_4444, onemax_4489, onemax_4624, onemax_4761, onemax_4900, \
onemax_4903, onemax_5000, onemax_5041, onemax_5184, onemax_5329, \
onemax_5461, onemax_5476, onemax_5555, onemax_5625, onemax_5776, \
onemax_5929, onemax_6000, onemax_6084, onemax_6144, onemax_6241, \
onemax_6400, onemax_6547, onemax_6561, onemax_6666, onemax_6724, \
onemax_6889, onemax_7000, onemax_7056, onemax_7225, onemax_7396, \
onemax_7569, onemax_7744, onemax_7777, onemax_7921, onemax_8000, \
onemax_8100, onemax_8192]

        """
        return cast(Iterator[Callable[[], "OneMax"]],
                    super().default_instances(  # type: ignore
                        scale_min, scale_max))
