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
4. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
   Fitness Assignment: Optimization without Bias for Good Solutions can be
   Efficient. *IEEE Transactions on Evolutionary Computation (TEVC)*. 2022.
   Early Access. https://dx.doi.org/10.1109/TEVC.2022.3191698
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
