"""
The one-dimensional Ising problem.

The one-dimensional Ising problem describes a ring. For each bit that differs
from its (right-side) neighboring bit, a penalty of 1 is incurred. The optimum
is a bit string of either all ones or all zeros. The optimal objective value
is 0, the worst-possible one is `n`.

1. Simon Fischer and Ingo Wegener. The one-dimensional Ising model: Mutation
   versus recombination. *Theoretical Computer Science.* 344(2-3):208-225.
   November 2005. doi: https://doi.org/10.1016/j.tcs.2005.04.002
2. Carola Doerr and Furong Ye and Naama Horesh and Hao Wang and Ofer M. Shir
   and Thomas Bäck. Benchmarking Discrete Optimization Heuristics with
   IOHprofiler. Applied Soft Computing 88:106027, March 2020,
   doi: https://doi.org/10.1016/j.asoc.2019.106027.
3. Clarissa Van Hoyweghen, David Edward Goldberg, and Bart Naudts. From Twomax
   To The Ising Model: Easy And Hard Symmetrical Problems. *In Proceedings of
   the Genetic and Evolutionary Computation Conference (GECCO'02),* July 9-13,
   2002, New York, NY, USA, pages 626-633. Morgan Kaufmann.
   http://gpbib.cs.ucl.ac.uk/gecco2002/GA013.pdf
4. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
   Fitness Assignment: Optimization without Bias for Good Solutions can be
   Efficient. *IEEE Transactions on Evolutionary Computation (TEVC)*. 2022.
   Early Access. https://dx.doi.org/10.1109/TEVC.2022.3191698
"""

import numba  # type: ignore
import numpy as np

from moptipy.examples.bitstrings.bitstring_problem import BitStringProblem


@numba.njit(nogil=True, cache=True)
def ising1d(x: np.ndarray) -> int:
    """
    Compute the objective value of the 1-dimensional Ising problem.

    :param x: the np array
    :return: the ising1d function value

    >>> ising1d(np.array([True, True, True, True, True]))
    0
    >>> ising1d(np.array([False, False, False, False, False]))
    0
    >>> ising1d(np.array([False, False, False, True, False]))
    2
    >>> ising1d(np.array([True, False, False, False, False]))
    2
    >>> ising1d(np.array([False, False, False, False, True]))
    2
    >>> ising1d(np.array([True, False, False, False, True]))
    2
    >>> ising1d(np.array([True, False, True, False, False]))
    4
    >>> ising1d(np.array([True, False, True, False, True, False]))
    6

    # n = 1 and 0 true bits
    >>> ising1d(np.array([0]))
    0

    # n = 1 and 1 true bit
    >>> ising1d(np.array([1]))
    0

    # n = 2 and 0 true bits
    >>> ising1d(np.array([0, 0]))
    0

    # n = 2 and 1 true bit
    >>> ising1d(np.array([1, 0]))
    2

    # n = 2 and 1 true bit
    >>> ising1d(np.array([0, 1]))
    2

    # n = 2 and 1 true bit
    >>> ising1d(np.array([0, 1]))
    2

    # n = 2 and 2 true bits
    >>> ising1d(np.array([1, 1]))
    0

    # n = 3 and 0 true bits
    >>> ising1d(np.array([0, 0, 0]))
    0

    # n = 3 and 1 true bit
    >>> ising1d(np.array([1, 0, 0]))
    2

    # n = 3 and 1 true bit
    >>> ising1d(np.array([0, 0, 1]))
    2

    # n = 3 and 1 true bit
    >>> ising1d(np.array([1, 0, 0]))
    2

    # n = 3 and 2 true bits
    >>> ising1d(np.array([1, 0, 1]))
    2

    # n = 3 and 2 true bits
    >>> ising1d(np.array([1, 0, 1]))
    2

    # n = 3 and 2 true bits
    >>> ising1d(np.array([1, 1, 0]))
    2

    # n = 3 and 3 true bits
    >>> ising1d(np.array([1, 1, 1]))
    0

    # n = 4 and 0 true bits
    >>> ising1d(np.array([0, 0, 0, 0]))
    0

    # n = 4 and 1 true bit
    >>> ising1d(np.array([1, 0, 0, 0]))
    2

    # n = 4 and 1 true bit
    >>> ising1d(np.array([0, 0, 1, 0]))
    2

    # n = 4 and 1 true bit
    >>> ising1d(np.array([1, 0, 0, 0]))
    2

    # n = 4 and 2 true bits
    >>> ising1d(np.array([1, 0, 0, 1]))
    2

    # n = 4 and 2 true bits
    >>> ising1d(np.array([1, 1, 0, 0]))
    2

    # n = 4 and 2 true bits
    >>> ising1d(np.array([0, 1, 1, 0]))
    2

    # n = 4 and 3 true bits
    >>> ising1d(np.array([0, 1, 1, 1]))
    2

    # n = 4 and 3 true bits
    >>> ising1d(np.array([1, 1, 0, 1]))
    2

    # n = 4 and 3 true bits
    >>> ising1d(np.array([0, 1, 1, 1]))
    2

    # n = 4 and 4 true bits
    >>> ising1d(np.array([1, 1, 1, 1]))
    0

    # n = 5 and 0 true bits
    >>> ising1d(np.array([0, 0, 0, 0, 0]))
    0

    # n = 5 and 1 true bit
    >>> ising1d(np.array([0, 0, 0, 0, 1]))
    2

    # n = 5 and 1 true bit
    >>> ising1d(np.array([0, 0, 0, 0, 1]))
    2

    # n = 5 and 1 true bit
    >>> ising1d(np.array([0, 1, 0, 0, 0]))
    2

    # n = 5 and 2 true bits
    >>> ising1d(np.array([0, 1, 0, 0, 1]))
    4

    # n = 5 and 2 true bits
    >>> ising1d(np.array([0, 1, 0, 0, 1]))
    4

    # n = 5 and 2 true bits
    >>> ising1d(np.array([0, 1, 1, 0, 0]))
    2

    # n = 5 and 3 true bits
    >>> ising1d(np.array([1, 1, 0, 1, 0]))
    4

    # n = 5 and 3 true bits
    >>> ising1d(np.array([1, 1, 1, 0, 0]))
    2

    # n = 5 and 3 true bits
    >>> ising1d(np.array([1, 0, 1, 0, 1]))
    4

    # n = 5 and 4 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 0]))
    2

    # n = 5 and 4 true bits
    >>> ising1d(np.array([1, 1, 0, 1, 1]))
    2

    # n = 5 and 4 true bits
    >>> ising1d(np.array([1, 1, 0, 1, 1]))
    2

    # n = 5 and 5 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1]))
    0

    # n = 6 and 0 true bits
    >>> ising1d(np.array([0, 0, 0, 0, 0, 0]))
    0

    # n = 6 and 1 true bit
    >>> ising1d(np.array([0, 0, 0, 0, 0, 1]))
    2

    # n = 6 and 1 true bit
    >>> ising1d(np.array([0, 0, 0, 0, 1, 0]))
    2

    # n = 6 and 1 true bit
    >>> ising1d(np.array([0, 1, 0, 0, 0, 0]))
    2

    # n = 6 and 2 true bits
    >>> ising1d(np.array([1, 1, 0, 0, 0, 0]))
    2

    # n = 6 and 2 true bits
    >>> ising1d(np.array([1, 1, 0, 0, 0, 0]))
    2

    # n = 6 and 2 true bits
    >>> ising1d(np.array([0, 0, 0, 1, 1, 0]))
    2

    # n = 6 and 3 true bits
    >>> ising1d(np.array([1, 0, 0, 1, 0, 1]))
    4

    # n = 6 and 3 true bits
    >>> ising1d(np.array([1, 0, 0, 0, 1, 1]))
    2

    # n = 6 and 3 true bits
    >>> ising1d(np.array([1, 1, 0, 0, 1, 0]))
    4

    # n = 6 and 4 true bits
    >>> ising1d(np.array([1, 0, 1, 1, 1, 0]))
    4

    # n = 6 and 4 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 0, 0]))
    2

    # n = 6 and 4 true bits
    >>> ising1d(np.array([1, 1, 0, 1, 0, 1]))
    4

    # n = 6 and 5 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 0, 1]))
    2

    # n = 6 and 5 true bits
    >>> ising1d(np.array([1, 0, 1, 1, 1, 1]))
    2

    # n = 6 and 5 true bits
    >>> ising1d(np.array([0, 1, 1, 1, 1, 1]))
    2

    # n = 6 and 6 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 1]))
    0

    # n = 7 and 0 true bits
    >>> ising1d(np.array([0, 0, 0, 0, 0, 0, 0]))
    0

    # n = 7 and 1 true bit
    >>> ising1d(np.array([0, 0, 0, 1, 0, 0, 0]))
    2

    # n = 7 and 1 true bit
    >>> ising1d(np.array([0, 0, 0, 0, 0, 1, 0]))
    2

    # n = 7 and 1 true bit
    >>> ising1d(np.array([0, 0, 0, 0, 0, 1, 0]))
    2

    # n = 7 and 2 true bits
    >>> ising1d(np.array([0, 1, 1, 0, 0, 0, 0]))
    2

    # n = 7 and 2 true bits
    >>> ising1d(np.array([1, 0, 0, 0, 0, 1, 0]))
    4

    # n = 7 and 2 true bits
    >>> ising1d(np.array([1, 0, 0, 0, 0, 0, 1]))
    2

    # n = 7 and 3 true bits
    >>> ising1d(np.array([1, 0, 0, 0, 0, 1, 1]))
    2

    # n = 7 and 3 true bits
    >>> ising1d(np.array([1, 0, 1, 0, 0, 1, 0]))
    6

    # n = 7 and 3 true bits
    >>> ising1d(np.array([1, 0, 0, 1, 0, 1, 0]))
    6

    # n = 7 and 4 true bits
    >>> ising1d(np.array([0, 1, 1, 0, 1, 1, 0]))
    4

    # n = 7 and 4 true bits
    >>> ising1d(np.array([0, 1, 0, 1, 1, 0, 1]))
    6

    # n = 7 and 4 true bits
    >>> ising1d(np.array([1, 1, 1, 0, 1, 0, 0]))
    4

    # n = 7 and 5 true bits
    >>> ising1d(np.array([1, 0, 1, 1, 1, 0, 1]))
    4

    # n = 7 and 5 true bits
    >>> ising1d(np.array([0, 1, 1, 1, 0, 1, 1]))
    4

    # n = 7 and 5 true bits
    >>> ising1d(np.array([1, 1, 1, 0, 1, 1, 0]))
    4

    # n = 7 and 6 true bits
    >>> ising1d(np.array([1, 1, 1, 0, 1, 1, 1]))
    2

    # n = 7 and 6 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 1, 0]))
    2

    # n = 7 and 6 true bits
    >>> ising1d(np.array([0, 1, 1, 1, 1, 1, 1]))
    2

    # n = 7 and 7 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 8 and 0 true bits
    >>> ising1d(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    0

    # n = 8 and 1 true bit
    >>> ising1d(np.array([0, 0, 1, 0, 0, 0, 0, 0]))
    2

    # n = 8 and 1 true bit
    >>> ising1d(np.array([0, 0, 1, 0, 0, 0, 0, 0]))
    2

    # n = 8 and 1 true bit
    >>> ising1d(np.array([0, 0, 0, 0, 0, 1, 0, 0]))
    2

    # n = 8 and 2 true bits
    >>> ising1d(np.array([1, 0, 1, 0, 0, 0, 0, 0]))
    4

    # n = 8 and 2 true bits
    >>> ising1d(np.array([0, 0, 0, 0, 1, 1, 0, 0]))
    2

    # n = 8 and 2 true bits
    >>> ising1d(np.array([0, 0, 0, 1, 0, 0, 0, 1]))
    4

    # n = 8 and 3 true bits
    >>> ising1d(np.array([0, 0, 1, 1, 1, 0, 0, 0]))
    2

    # n = 8 and 3 true bits
    >>> ising1d(np.array([1, 0, 1, 0, 0, 0, 1, 0]))
    6

    # n = 8 and 3 true bits
    >>> ising1d(np.array([0, 1, 0, 1, 0, 0, 1, 0]))
    6

    # n = 8 and 4 true bits
    >>> ising1d(np.array([0, 1, 0, 1, 1, 0, 0, 1]))
    6

    # n = 8 and 4 true bits
    >>> ising1d(np.array([1, 0, 1, 0, 1, 0, 1, 0]))
    8

    # n = 8 and 4 true bits
    >>> ising1d(np.array([1, 0, 1, 0, 1, 0, 0, 1]))
    6

    # n = 8 and 5 true bits
    >>> ising1d(np.array([1, 0, 1, 0, 0, 1, 1, 1]))
    4

    # n = 8 and 5 true bits
    >>> ising1d(np.array([1, 1, 0, 1, 0, 0, 1, 1]))
    4

    # n = 8 and 5 true bits
    >>> ising1d(np.array([1, 0, 0, 1, 0, 1, 1, 1]))
    4

    # n = 8 and 6 true bits
    >>> ising1d(np.array([0, 1, 1, 1, 0, 1, 1, 1]))
    4

    # n = 8 and 6 true bits
    >>> ising1d(np.array([0, 0, 1, 1, 1, 1, 1, 1]))
    2

    # n = 8 and 6 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 1, 0, 0]))
    2

    # n = 8 and 7 true bits
    >>> ising1d(np.array([1, 1, 1, 0, 1, 1, 1, 1]))
    2

    # n = 8 and 7 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 1, 0, 1]))
    2

    # n = 8 and 7 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 1, 0, 1]))
    2

    # n = 8 and 8 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 9 and 0 true bits
    >>> ising1d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
    0

    # n = 9 and 1 true bit
    >>> ising1d(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]))
    2

    # n = 9 and 1 true bit
    >>> ising1d(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]))
    2

    # n = 9 and 1 true bit
    >>> ising1d(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]))
    2

    # n = 9 and 2 true bits
    >>> ising1d(np.array([0, 1, 0, 1, 0, 0, 0, 0, 0]))
    4

    # n = 9 and 2 true bits
    >>> ising1d(np.array([0, 1, 0, 0, 1, 0, 0, 0, 0]))
    4

    # n = 9 and 2 true bits
    >>> ising1d(np.array([0, 1, 0, 1, 0, 0, 0, 0, 0]))
    4

    # n = 9 and 3 true bits
    >>> ising1d(np.array([0, 0, 1, 0, 0, 1, 0, 1, 0]))
    6

    # n = 9 and 3 true bits
    >>> ising1d(np.array([0, 0, 0, 1, 1, 0, 0, 1, 0]))
    4

    # n = 9 and 3 true bits
    >>> ising1d(np.array([0, 0, 0, 1, 0, 0, 0, 1, 1]))
    4

    # n = 9 and 4 true bits
    >>> ising1d(np.array([1, 0, 1, 1, 1, 0, 0, 0, 0]))
    4

    # n = 9 and 4 true bits
    >>> ising1d(np.array([1, 1, 1, 0, 1, 0, 0, 0, 0]))
    4

    # n = 9 and 4 true bits
    >>> ising1d(np.array([0, 1, 0, 0, 0, 1, 1, 0, 1]))
    6

    # n = 9 and 5 true bits
    >>> ising1d(np.array([0, 0, 1, 1, 0, 0, 1, 1, 1]))
    4

    # n = 9 and 5 true bits
    >>> ising1d(np.array([0, 1, 0, 1, 1, 0, 1, 0, 1]))
    8

    # n = 9 and 5 true bits
    >>> ising1d(np.array([1, 0, 1, 1, 1, 0, 0, 1, 0]))
    6

    # n = 9 and 6 true bits
    >>> ising1d(np.array([1, 0, 1, 1, 1, 1, 1, 0, 0]))
    4

    # n = 9 and 6 true bits
    >>> ising1d(np.array([0, 0, 1, 1, 1, 1, 1, 1, 0]))
    2

    # n = 9 and 6 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 0, 1, 0, 0, 1]))
    4

    # n = 9 and 7 true bits
    >>> ising1d(np.array([1, 1, 0, 1, 1, 0, 1, 1, 1]))
    4

    # n = 9 and 7 true bits
    >>> ising1d(np.array([1, 1, 0, 1, 1, 1, 1, 1, 0]))
    4

    # n = 9 and 7 true bits
    >>> ising1d(np.array([1, 0, 1, 0, 1, 1, 1, 1, 1]))
    4

    # n = 9 and 8 true bits
    >>> ising1d(np.array([1, 0, 1, 1, 1, 1, 1, 1, 1]))
    2

    # n = 9 and 8 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 1, 0, 1, 1]))
    2

    # n = 9 and 8 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 0, 1, 1, 1]))
    2

    # n = 9 and 9 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 10 and 0 true bits
    >>> ising1d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    0

    # n = 10 and 1 true bit
    >>> ising1d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    2

    # n = 10 and 1 true bit
    >>> ising1d(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
    2

    # n = 10 and 1 true bit
    >>> ising1d(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    2

    # n = 10 and 2 true bits
    >>> ising1d(np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0]))
    4

    # n = 10 and 2 true bits
    >>> ising1d(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    2

    # n = 10 and 2 true bits
    >>> ising1d(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1]))
    4

    # n = 10 and 3 true bits
    >>> ising1d(np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]))
    2

    # n = 10 and 3 true bits
    >>> ising1d(np.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 0]))
    6

    # n = 10 and 3 true bits
    >>> ising1d(np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 1]))
    6

    # n = 10 and 4 true bits
    >>> ising1d(np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 0]))
    6

    # n = 10 and 4 true bits
    >>> ising1d(np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1]))
    6

    # n = 10 and 4 true bits
    >>> ising1d(np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 1]))
    4

    # n = 10 and 5 true bits
    >>> ising1d(np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 0]))
    6

    # n = 10 and 5 true bits
    >>> ising1d(np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0]))
    8

    # n = 10 and 5 true bits
    >>> ising1d(np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1]))
    6

    # n = 10 and 6 true bits
    >>> ising1d(np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 1]))
    4

    # n = 10 and 6 true bits
    >>> ising1d(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 0]))
    6

    # n = 10 and 6 true bits
    >>> ising1d(np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1]))
    4

    # n = 10 and 7 true bits
    >>> ising1d(np.array([1, 0, 0, 0, 1, 1, 1, 1, 1, 1]))
    2

    # n = 10 and 7 true bits
    >>> ising1d(np.array([1, 0, 0, 1, 1, 1, 1, 1, 1, 0]))
    4

    # n = 10 and 7 true bits
    >>> ising1d(np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 1]))
    6

    # n = 10 and 8 true bits
    >>> ising1d(np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1]))
    4

    # n = 10 and 8 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 0]))
    4

    # n = 10 and 8 true bits
    >>> ising1d(np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1]))
    4

    # n = 10 and 9 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1]))
    2

    # n = 10 and 9 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1]))
    2

    # n = 10 and 9 true bits
    >>> ising1d(np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1]))
    2

    # n = 10 and 10 true bits
    >>> ising1d(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    0
    """
    n: int = 0
    prev: bool = x[-1]
    for cur in x:
        if cur != prev:
            n += 1
        prev = cur
    return n


class Ising1d(BitStringProblem):
    """The one-dimensional Ising problem."""

    def __init__(self, n: int) -> None:  # +book
        """
        Initialize the one-dimensional Ising problem.

        :param n: the dimension of the problem

        >>> Ising1d(7).n
        7
        >>> Ising1d(3).evaluate(np.array([True, False, True]))
        2
        """
        super().__init__(n)
        self.evaluate = ising1d  # type: ignore

    def __str__(self) -> str:
        """
        Get the name of the one-dimensional Ising problem.

        :return: `ising1d_` + length of string

        >>> Ising1d(5)
        ising1d_5
        """
        return f"ising1d_{self.n}"
