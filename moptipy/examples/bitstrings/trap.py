"""
The well-known Trap problem.

The Trap function is similar to OneMax, but swaps the worst possible solution
with the global optimum. This means that the best-possible objective value, 0,
is reached for the string of all `False` bits. The worst objective value, `n`,
is reached by the strings with exactly one `True` bit.

1. Stefan Droste, Thomas Jansen, and Ingo Wegener. On the Analysis of the
   (1+1) Evolutionary Algorithm. *Theoretical Computer Science.*
   276(1-2):51-81. April 2002.
   doi: https://doi.org/10.1016/S0304-3975(01)00182-7
2. Siegfried Nijssen and Thomas Bäck. An Analysis of the Behavior of
   Simplified Evolutionary Algorithms on Trap Functions. IEEE Transactions on
   Evolutionary Computation. 7(1):11-22. 2003.
   doi: https://doi.org/10.1109/TEVC.2002.806169
3. Thomas Weise, Zhize Wu, Xinlu Li, and Yan Chen. Frequency Fitness
   Assignment: Making Optimization Algorithms Invariant under Bijective
   Transformations of the Objective Function Value. *IEEE Transactions on
   Evolutionary Computation* 25(2):307-319. April 2021. Preprint available at
   arXiv:2001.01416v5 [cs.NE] 15 Oct 2020.
   https://dx.doi.org/10.1109/TEVC.2020.3032090
4. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
   Fitness Assignment: Optimization without Bias for Good Solutions can be
   Efficient. *IEEE Transactions on Evolutionary Computation (TEVC)*.
   27(4):980-992. August 2023.
   doi: https://doi.org/10.1109/TEVC.2022.3191698
"""
from typing import Callable, Final, Iterator, cast

import numba  # type: ignore
import numpy as np

from moptipy.examples.bitstrings.bitstring_problem import BitStringProblem


@numba.njit(nogil=True, cache=True)
def trap(x: np.ndarray) -> int:
    """
    Compute the trap objective value.

    :param x: the np array
    :return: the trap function value

    >>> print(trap(np.array([True, True, False, False, False])))
    4
    >>> print(trap(np.array([True, False, True, False, False])))
    4
    >>> print(trap(np.array([False, True,  True, False, False])))
    4
    >>> print(trap(np.array([True, True, True, True, True])))
    1
    >>> print(trap(np.array([False, True, True, True, True])))
    2
    >>> print(trap(np.array([False, False, False, False, False])))
    0
    >>> print(trap(np.array([False, True,  False, False, False])))
    5
    >>> print(trap(np.array([False, True,  True, True, False])))
    3

    # n = 1 and 0 true bits
    >>> trap(np.array([0]))
    0

    # n = 1 and 1 true bit
    >>> trap(np.array([1]))
    1

    # n = 2 and 0 true bits
    >>> trap(np.array([0, 0]))
    0

    # n = 2 and 1 true bit
    >>> trap(np.array([0, 1]))
    2

    # n = 2 and 1 true bit
    >>> trap(np.array([0, 1]))
    2

    # n = 2 and 1 true bit
    >>> trap(np.array([0, 1]))
    2

    # n = 2 and 2 true bits
    >>> trap(np.array([1, 1]))
    1

    # n = 3 and 0 true bits
    >>> trap(np.array([0, 0, 0]))
    0

    # n = 3 and 1 true bit
    >>> trap(np.array([0, 1, 0]))
    3

    # n = 3 and 1 true bit
    >>> trap(np.array([1, 0, 0]))
    3

    # n = 3 and 1 true bit
    >>> trap(np.array([0, 0, 1]))
    3

    # n = 3 and 2 true bits
    >>> trap(np.array([0, 1, 1]))
    2

    # n = 3 and 2 true bits
    >>> trap(np.array([1, 0, 1]))
    2

    # n = 3 and 2 true bits
    >>> trap(np.array([0, 1, 1]))
    2

    # n = 3 and 3 true bits
    >>> trap(np.array([1, 1, 1]))
    1

    # n = 4 and 0 true bits
    >>> trap(np.array([0, 0, 0, 0]))
    0

    # n = 4 and 1 true bit
    >>> trap(np.array([1, 0, 0, 0]))
    4

    # n = 4 and 1 true bit
    >>> trap(np.array([1, 0, 0, 0]))
    4

    # n = 4 and 1 true bit
    >>> trap(np.array([0, 0, 0, 1]))
    4

    # n = 4 and 2 true bits
    >>> trap(np.array([1, 1, 0, 0]))
    3

    # n = 4 and 2 true bits
    >>> trap(np.array([0, 0, 1, 1]))
    3

    # n = 4 and 2 true bits
    >>> trap(np.array([0, 1, 1, 0]))
    3

    # n = 4 and 3 true bits
    >>> trap(np.array([0, 1, 1, 1]))
    2

    # n = 4 and 3 true bits
    >>> trap(np.array([1, 1, 1, 0]))
    2

    # n = 4 and 3 true bits
    >>> trap(np.array([1, 1, 0, 1]))
    2

    # n = 4 and 4 true bits
    >>> trap(np.array([1, 1, 1, 1]))
    1

    # n = 5 and 0 true bits
    >>> trap(np.array([0, 0, 0, 0, 0]))
    0

    # n = 5 and 1 true bit
    >>> trap(np.array([0, 0, 0, 0, 1]))
    5

    # n = 5 and 1 true bit
    >>> trap(np.array([0, 0, 0, 1, 0]))
    5

    # n = 5 and 1 true bit
    >>> trap(np.array([1, 0, 0, 0, 0]))
    5

    # n = 5 and 2 true bits
    >>> trap(np.array([1, 0, 1, 0, 0]))
    4

    # n = 5 and 2 true bits
    >>> trap(np.array([1, 0, 1, 0, 0]))
    4

    # n = 5 and 2 true bits
    >>> trap(np.array([0, 1, 0, 0, 1]))
    4

    # n = 5 and 3 true bits
    >>> trap(np.array([1, 0, 1, 1, 0]))
    3

    # n = 5 and 3 true bits
    >>> trap(np.array([0, 1, 1, 0, 1]))
    3

    # n = 5 and 3 true bits
    >>> trap(np.array([0, 1, 1, 0, 1]))
    3

    # n = 5 and 4 true bits
    >>> trap(np.array([1, 0, 1, 1, 1]))
    2

    # n = 5 and 4 true bits
    >>> trap(np.array([1, 1, 1, 0, 1]))
    2

    # n = 5 and 4 true bits
    >>> trap(np.array([1, 1, 0, 1, 1]))
    2

    # n = 5 and 5 true bits
    >>> trap(np.array([1, 1, 1, 1, 1]))
    1

    # n = 6 and 0 true bits
    >>> trap(np.array([0, 0, 0, 0, 0, 0]))
    0

    # n = 6 and 1 true bit
    >>> trap(np.array([0, 0, 0, 0, 1, 0]))
    6

    # n = 6 and 1 true bit
    >>> trap(np.array([0, 0, 0, 0, 0, 1]))
    6

    # n = 6 and 1 true bit
    >>> trap(np.array([0, 0, 1, 0, 0, 0]))
    6

    # n = 6 and 2 true bits
    >>> trap(np.array([1, 0, 0, 1, 0, 0]))
    5

    # n = 6 and 2 true bits
    >>> trap(np.array([1, 0, 0, 0, 1, 0]))
    5

    # n = 6 and 2 true bits
    >>> trap(np.array([1, 0, 0, 0, 1, 0]))
    5

    # n = 6 and 3 true bits
    >>> trap(np.array([0, 1, 1, 1, 0, 0]))
    4

    # n = 6 and 3 true bits
    >>> trap(np.array([1, 1, 0, 0, 0, 1]))
    4

    # n = 6 and 3 true bits
    >>> trap(np.array([0, 1, 1, 1, 0, 0]))
    4

    # n = 6 and 4 true bits
    >>> trap(np.array([1, 0, 0, 1, 1, 1]))
    3

    # n = 6 and 4 true bits
    >>> trap(np.array([1, 0, 0, 1, 1, 1]))
    3

    # n = 6 and 4 true bits
    >>> trap(np.array([0, 1, 1, 1, 1, 0]))
    3

    # n = 6 and 5 true bits
    >>> trap(np.array([0, 1, 1, 1, 1, 1]))
    2

    # n = 6 and 5 true bits
    >>> trap(np.array([1, 1, 1, 1, 0, 1]))
    2

    # n = 6 and 5 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 0]))
    2

    # n = 6 and 6 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 1]))
    1

    # n = 7 and 0 true bits
    >>> trap(np.array([0, 0, 0, 0, 0, 0, 0]))
    0

    # n = 7 and 1 true bit
    >>> trap(np.array([0, 0, 1, 0, 0, 0, 0]))
    7

    # n = 7 and 1 true bit
    >>> trap(np.array([0, 0, 0, 0, 0, 1, 0]))
    7

    # n = 7 and 1 true bit
    >>> trap(np.array([0, 0, 0, 1, 0, 0, 0]))
    7

    # n = 7 and 2 true bits
    >>> trap(np.array([0, 0, 1, 0, 0, 0, 1]))
    6

    # n = 7 and 2 true bits
    >>> trap(np.array([0, 0, 1, 0, 0, 1, 0]))
    6

    # n = 7 and 2 true bits
    >>> trap(np.array([1, 0, 1, 0, 0, 0, 0]))
    6

    # n = 7 and 3 true bits
    >>> trap(np.array([0, 0, 1, 1, 0, 1, 0]))
    5

    # n = 7 and 3 true bits
    >>> trap(np.array([1, 0, 0, 1, 1, 0, 0]))
    5

    # n = 7 and 3 true bits
    >>> trap(np.array([1, 1, 0, 0, 0, 0, 1]))
    5

    # n = 7 and 4 true bits
    >>> trap(np.array([0, 1, 0, 0, 1, 1, 1]))
    4

    # n = 7 and 4 true bits
    >>> trap(np.array([1, 1, 0, 0, 1, 1, 0]))
    4

    # n = 7 and 4 true bits
    >>> trap(np.array([1, 0, 1, 0, 0, 1, 1]))
    4

    # n = 7 and 5 true bits
    >>> trap(np.array([1, 1, 0, 1, 1, 1, 0]))
    3

    # n = 7 and 5 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 0, 0]))
    3

    # n = 7 and 5 true bits
    >>> trap(np.array([1, 1, 0, 1, 0, 1, 1]))
    3

    # n = 7 and 6 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 1, 0]))
    2

    # n = 7 and 6 true bits
    >>> trap(np.array([1, 1, 0, 1, 1, 1, 1]))
    2

    # n = 7 and 6 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 1, 0]))
    2

    # n = 7 and 7 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 1, 1]))
    1

    # n = 8 and 0 true bits
    >>> trap(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    0

    # n = 8 and 1 true bit
    >>> trap(np.array([0, 0, 0, 0, 1, 0, 0, 0]))
    8

    # n = 8 and 1 true bit
    >>> trap(np.array([0, 1, 0, 0, 0, 0, 0, 0]))
    8

    # n = 8 and 1 true bit
    >>> trap(np.array([0, 0, 1, 0, 0, 0, 0, 0]))
    8

    # n = 8 and 2 true bits
    >>> trap(np.array([1, 0, 0, 0, 0, 1, 0, 0]))
    7

    # n = 8 and 2 true bits
    >>> trap(np.array([0, 1, 0, 0, 1, 0, 0, 0]))
    7

    # n = 8 and 2 true bits
    >>> trap(np.array([0, 1, 0, 0, 1, 0, 0, 0]))
    7

    # n = 8 and 3 true bits
    >>> trap(np.array([0, 1, 0, 0, 0, 1, 1, 0]))
    6

    # n = 8 and 3 true bits
    >>> trap(np.array([0, 1, 1, 0, 0, 1, 0, 0]))
    6

    # n = 8 and 3 true bits
    >>> trap(np.array([1, 1, 0, 0, 1, 0, 0, 0]))
    6

    # n = 8 and 4 true bits
    >>> trap(np.array([1, 0, 1, 0, 1, 0, 0, 1]))
    5

    # n = 8 and 4 true bits
    >>> trap(np.array([0, 1, 1, 1, 1, 0, 0, 0]))
    5

    # n = 8 and 4 true bits
    >>> trap(np.array([1, 1, 1, 0, 0, 1, 0, 0]))
    5

    # n = 8 and 5 true bits
    >>> trap(np.array([1, 1, 0, 0, 0, 1, 1, 1]))
    4

    # n = 8 and 5 true bits
    >>> trap(np.array([1, 1, 1, 0, 0, 1, 1, 0]))
    4

    # n = 8 and 5 true bits
    >>> trap(np.array([0, 0, 1, 1, 0, 1, 1, 1]))
    4

    # n = 8 and 6 true bits
    >>> trap(np.array([0, 1, 1, 1, 1, 1, 1, 0]))
    3

    # n = 8 and 6 true bits
    >>> trap(np.array([1, 1, 0, 1, 1, 1, 0, 1]))
    3

    # n = 8 and 6 true bits
    >>> trap(np.array([1, 1, 1, 0, 0, 1, 1, 1]))
    3

    # n = 8 and 7 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 1, 0, 1]))
    2

    # n = 8 and 7 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 1, 1, 0]))
    2

    # n = 8 and 7 true bits
    >>> trap(np.array([1, 0, 1, 1, 1, 1, 1, 1]))
    2

    # n = 8 and 8 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    1

    # n = 9 and 0 true bits
    >>> trap(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
    0

    # n = 9 and 1 true bit
    >>> trap(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]))
    9

    # n = 9 and 1 true bit
    >>> trap(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]))
    9

    # n = 9 and 1 true bit
    >>> trap(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]))
    9

    # n = 9 and 2 true bits
    >>> trap(np.array([1, 0, 0, 0, 1, 0, 0, 0, 0]))
    8

    # n = 9 and 2 true bits
    >>> trap(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]))
    8

    # n = 9 and 2 true bits
    >>> trap(np.array([0, 0, 0, 1, 0, 0, 0, 1, 0]))
    8

    # n = 9 and 3 true bits
    >>> trap(np.array([0, 1, 0, 0, 0, 1, 0, 0, 1]))
    7

    # n = 9 and 3 true bits
    >>> trap(np.array([1, 0, 0, 1, 1, 0, 0, 0, 0]))
    7

    # n = 9 and 3 true bits
    >>> trap(np.array([1, 1, 0, 0, 0, 0, 1, 0, 0]))
    7

    # n = 9 and 4 true bits
    >>> trap(np.array([1, 0, 1, 1, 0, 0, 1, 0, 0]))
    6

    # n = 9 and 4 true bits
    >>> trap(np.array([1, 0, 1, 0, 0, 0, 1, 1, 0]))
    6

    # n = 9 and 4 true bits
    >>> trap(np.array([1, 0, 0, 1, 0, 0, 1, 1, 0]))
    6

    # n = 9 and 5 true bits
    >>> trap(np.array([1, 0, 1, 1, 0, 0, 0, 1, 1]))
    5

    # n = 9 and 5 true bits
    >>> trap(np.array([0, 1, 0, 1, 0, 1, 1, 0, 1]))
    5

    # n = 9 and 5 true bits
    >>> trap(np.array([1, 0, 0, 0, 1, 0, 1, 1, 1]))
    5

    # n = 9 and 6 true bits
    >>> trap(np.array([1, 1, 1, 0, 0, 1, 0, 1, 1]))
    4

    # n = 9 and 6 true bits
    >>> trap(np.array([0, 1, 1, 0, 1, 1, 0, 1, 1]))
    4

    # n = 9 and 6 true bits
    >>> trap(np.array([1, 0, 1, 0, 1, 1, 1, 0, 1]))
    4

    # n = 9 and 7 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]))
    3

    # n = 9 and 7 true bits
    >>> trap(np.array([1, 0, 1, 1, 1, 1, 1, 0, 1]))
    3

    # n = 9 and 7 true bits
    >>> trap(np.array([1, 1, 0, 1, 1, 1, 1, 0, 1]))
    3

    # n = 9 and 8 true bits
    >>> trap(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1]))
    2

    # n = 9 and 8 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 1, 1, 1, 0]))
    2

    # n = 9 and 8 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 0, 1, 1, 1]))
    2

    # n = 9 and 9 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
    1

    # n = 10 and 0 true bits
    >>> trap(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    0

    # n = 10 and 1 true bit
    >>> trap(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
    10

    # n = 10 and 1 true bit
    >>> trap(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
    10

    # n = 10 and 1 true bit
    >>> trap(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
    10

    # n = 10 and 2 true bits
    >>> trap(np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
    9

    # n = 10 and 2 true bits
    >>> trap(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1]))
    9

    # n = 10 and 2 true bits
    >>> trap(np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0]))
    9

    # n = 10 and 3 true bits
    >>> trap(np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 1]))
    8

    # n = 10 and 3 true bits
    >>> trap(np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1]))
    8

    # n = 10 and 3 true bits
    >>> trap(np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0]))
    8

    # n = 10 and 4 true bits
    >>> trap(np.array([0, 0, 1, 0, 1, 0, 0, 1, 1, 0]))
    7

    # n = 10 and 4 true bits
    >>> trap(np.array([1, 0, 0, 0, 0, 0, 1, 0, 1, 1]))
    7

    # n = 10 and 4 true bits
    >>> trap(np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0]))
    7

    # n = 10 and 5 true bits
    >>> trap(np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 1]))
    6

    # n = 10 and 5 true bits
    >>> trap(np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 1]))
    6

    # n = 10 and 5 true bits
    >>> trap(np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0]))
    6

    # n = 10 and 6 true bits
    >>> trap(np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1]))
    5

    # n = 10 and 6 true bits
    >>> trap(np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1]))
    5

    # n = 10 and 6 true bits
    >>> trap(np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1]))
    5

    # n = 10 and 7 true bits
    >>> trap(np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1]))
    4

    # n = 10 and 7 true bits
    >>> trap(np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 0]))
    4

    # n = 10 and 7 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 0]))
    4

    # n = 10 and 8 true bits
    >>> trap(np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1]))
    3

    # n = 10 and 8 true bits
    >>> trap(np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 1]))
    3

    # n = 10 and 8 true bits
    >>> trap(np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]))
    3

    # n = 10 and 9 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1]))
    2

    # n = 10 and 9 true bits
    >>> trap(np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1]))
    2

    # n = 10 and 9 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1]))
    2

    # n = 10 and 10 true bits
    >>> trap(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    1
    """
    n: Final[int] = len(x)
    res: Final[int] = int(x.sum())
    return 0 if res <= 0 else n - res + 1


class Trap(BitStringProblem):
    """The trap problem."""

    def __init__(self, n: int) -> None:
        """
        Initialize the trap objective function.

        :param n: the dimension of the problem

        >>> print(Trap(55).n)
        55
        >>> print(Trap(4).evaluate(np.array([True, True, False, True])))
        2
        """
        super().__init__(n)
        self.evaluate = trap  # type: ignore

    def __str__(self) -> str:
        """
        Get the name of the trap objective function.

        :return: `trap_` + length of string

        >>> print(Trap(33))
        trap_33
        """
        return f"trap_{self.n}"

    @classmethod
    def default_instances(
            cls: type, scale_min: int = 3, scale_max: int = 333) \
            -> Iterator[Callable[[], "Trap"]]:
        """
        Get the 77 default instances of the :class:`Trap` problem.

        :param scale_min: the minimum permitted scale, by default `3`
        :param scale_max: the maximum permitted scale, by default `333`
        :returns: a sequence of default :class:`Trap` instances

        >>> len(list(Trap.default_instances()))
        77

        >>> [x() for x in Trap.default_instances()]
        [trap_3, trap_4, trap_5, trap_6, trap_7, trap_8, trap_9, trap_10, \
trap_11, trap_12, trap_13, trap_14, trap_15, trap_16, trap_17, trap_18, \
trap_19, trap_20, trap_21, trap_22, trap_23, trap_24, trap_25, trap_26, \
trap_27, trap_28, trap_29, trap_30, trap_31, trap_32, trap_33, trap_36, \
trap_40, trap_41, trap_42, trap_44, trap_48, trap_49, trap_50, trap_55, \
trap_59, trap_60, trap_64, trap_66, trap_70, trap_77, trap_79, trap_80, \
trap_81, trap_85, trap_88, trap_90, trap_96, trap_99, trap_100, trap_107, \
trap_111, trap_121, trap_125, trap_128, trap_144, trap_149, trap_169, \
trap_170, trap_192, trap_196, trap_199, trap_200, trap_222, trap_225, \
trap_243, trap_256, trap_269, trap_289, trap_300, trap_324, trap_333]
        """
        return cast("Iterator[Callable[[], Trap]]",
                    super().default_instances(  # type: ignore
                        scale_min, scale_max))
