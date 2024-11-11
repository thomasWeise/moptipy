"""
The Plateau problem.

The plateau problem is basically OneMax, but with a neutral region of `k` bit
flips before the optimum. The best objective value, 0, is reached if all bits
are `True`. The worst objective value `n`, is reached if all bits are `False`.

1. Denis Antipov and Benjamin Doerr. Precise Runtime Analysis for Plateaus.
   Parallel Problem Solving from Nature (PPSN XV), 2018, Part II. LNCS 11102,
   pp. 117-128.
   doi: https://doi.org/10.1007/978-3-319-99259-4_10
2. Thomas Weise, Zhize Wu, Xinlu Li, and Yan Chen. Frequency Fitness
   Assignment: Making Optimization Algorithms Invariant under Bijective
   Transformations of the Objective Function Value. *IEEE Transactions on
   Evolutionary Computation* 25(2):307-319. April 2021. Preprint available at
   arXiv:2001.01416v5 [cs.NE] 15 Oct 2020.
   https://dx.doi.org/10.1109/TEVC.2020.3032090
3. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
   Fitness Assignment: Optimization without Bias for Good Solutions can be
   Efficient. *IEEE Transactions on Evolutionary Computation (TEVC)*. 2022.
   Early Access. https://dx.doi.org/10.1109/TEVC.2022.3191698

This is code is part of the research work of Mr. Jiazheng ZENG (曾嘉政),
a Master's student at the Institute of Applied Optimization
(应用优化研究所, http://iao.hfuu.edu.cn) of the School of Artificial
Intelligence and Big Data (人工智能与大数据学院) at
Hefei University (合肥大学) in
Hefei, Anhui, China (中国安徽省合肥市) under the supervision of
Prof. Dr. Thomas Weise (汤卫思教授).
"""

from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.examples.bitstrings.bitstring_problem import BitStringNKProblem


@numba.njit(nogil=True, cache=True)
def plateau(x: np.ndarray, k: int) -> int:
    """
    Compute the plateau value.

    :param x: the np array
    :param k: the k parameter
    :return: plateau value

    # n = 6, k = 2, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0]), 2)
    6

    # n = 6, k = 2, and 1 true bit
    >>> plateau(np.array([0, 0, 0, 1, 0, 0]), 2)
    5

    # n = 6, k = 2, and 2 true bits
    >>> plateau(np.array([0, 0, 1, 0, 1, 0]), 2)
    4

    # n = 6, k = 2, and 3 true bits
    >>> plateau(np.array([1, 1, 1, 0, 0, 0]), 2)
    3

    # n = 6, k = 2, and 4 true bits
    >>> plateau(np.array([1, 1, 1, 1, 0, 0]), 2)
    2

    # n = 6, k = 2, and 5 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 0]), 2)
    2

    # n = 6, k = 2, and 6 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 7, k = 2, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0]), 2)
    7

    # n = 7, k = 2, and 1 true bit
    >>> plateau(np.array([0, 0, 1, 0, 0, 0, 0]), 2)
    6

    # n = 7, k = 2, and 2 true bits
    >>> plateau(np.array([0, 1, 0, 1, 0, 0, 0]), 2)
    5

    # n = 7, k = 2, and 3 true bits
    >>> plateau(np.array([0, 1, 0, 0, 1, 0, 1]), 2)
    4

    # n = 7, k = 2, and 4 true bits
    >>> plateau(np.array([1, 0, 0, 0, 1, 1, 1]), 2)
    3

    # n = 7, k = 2, and 5 true bits
    >>> plateau(np.array([1, 0, 1, 1, 0, 1, 1]), 2)
    2

    # n = 7, k = 2, and 6 true bits
    >>> plateau(np.array([1, 1, 1, 1, 0, 1, 1]), 2)
    2

    # n = 7, k = 2, and 7 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 8, k = 2, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0]), 2)
    8

    # n = 8, k = 2, and 1 true bit
    >>> plateau(np.array([0, 1, 0, 0, 0, 0, 0, 0]), 2)
    7

    # n = 8, k = 2, and 2 true bits
    >>> plateau(np.array([1, 0, 0, 0, 1, 0, 0, 0]), 2)
    6

    # n = 8, k = 2, and 3 true bits
    >>> plateau(np.array([0, 0, 0, 1, 0, 1, 0, 1]), 2)
    5

    # n = 8, k = 2, and 4 true bits
    >>> plateau(np.array([0, 0, 0, 1, 1, 1, 0, 1]), 2)
    4

    # n = 8, k = 2, and 5 true bits
    >>> plateau(np.array([1, 1, 0, 1, 1, 0, 0, 1]), 2)
    3

    # n = 8, k = 2, and 6 true bits
    >>> plateau(np.array([0, 0, 1, 1, 1, 1, 1, 1]), 2)
    2

    # n = 8, k = 2, and 7 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 0, 1, 1]), 2)
    2

    # n = 8, k = 2, and 8 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 8, k = 3, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0]), 3)
    8

    # n = 8, k = 3, and 1 true bit
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 1]), 3)
    7

    # n = 8, k = 3, and 2 true bits
    >>> plateau(np.array([0, 0, 1, 0, 0, 0, 0, 1]), 3)
    6

    # n = 8, k = 3, and 3 true bits
    >>> plateau(np.array([0, 1, 0, 0, 1, 1, 0, 0]), 3)
    5

    # n = 8, k = 3, and 4 true bits
    >>> plateau(np.array([1, 0, 0, 0, 1, 1, 0, 1]), 3)
    4

    # n = 8, k = 3, and 5 true bits
    >>> plateau(np.array([1, 1, 0, 0, 1, 0, 1, 1]), 3)
    3

    # n = 8, k = 3, and 6 true bits
    >>> plateau(np.array([1, 0, 1, 1, 1, 0, 1, 1]), 3)
    3

    # n = 8, k = 3, and 7 true bits
    >>> plateau(np.array([0, 1, 1, 1, 1, 1, 1, 1]), 3)
    3

    # n = 8, k = 3, and 8 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1]), 3)
    0

    # n = 9, k = 2, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), 2)
    9

    # n = 9, k = 2, and 1 true bit
    >>> plateau(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]), 2)
    8

    # n = 9, k = 2, and 2 true bits
    >>> plateau(np.array([0, 0, 0, 0, 1, 1, 0, 0, 0]), 2)
    7

    # n = 9, k = 2, and 3 true bits
    >>> plateau(np.array([0, 1, 0, 1, 0, 1, 0, 0, 0]), 2)
    6

    # n = 9, k = 2, and 4 true bits
    >>> plateau(np.array([1, 1, 1, 0, 0, 0, 0, 0, 1]), 2)
    5

    # n = 9, k = 2, and 5 true bits
    >>> plateau(np.array([1, 0, 1, 1, 1, 0, 0, 1, 0]), 2)
    4

    # n = 9, k = 2, and 6 true bits
    >>> plateau(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1]), 2)
    3

    # n = 9, k = 2, and 7 true bits
    >>> plateau(np.array([1, 1, 1, 0, 1, 1, 0, 1, 1]), 2)
    2

    # n = 9, k = 2, and 8 true bits
    >>> plateau(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1]), 2)
    2

    # n = 9, k = 2, and 9 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 9, k = 3, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), 3)
    9

    # n = 9, k = 3, and 1 true bit
    >>> plateau(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]), 3)
    8

    # n = 9, k = 3, and 2 true bits
    >>> plateau(np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]), 3)
    7

    # n = 9, k = 3, and 3 true bits
    >>> plateau(np.array([0, 0, 0, 0, 1, 0, 0, 1, 1]), 3)
    6

    # n = 9, k = 3, and 4 true bits
    >>> plateau(np.array([0, 1, 1, 0, 0, 1, 0, 0, 1]), 3)
    5

    # n = 9, k = 3, and 5 true bits
    >>> plateau(np.array([1, 1, 0, 1, 1, 1, 0, 0, 0]), 3)
    4

    # n = 9, k = 3, and 6 true bits
    >>> plateau(np.array([0, 1, 1, 0, 1, 1, 1, 1, 0]), 3)
    3

    # n = 9, k = 3, and 7 true bits
    >>> plateau(np.array([1, 1, 1, 0, 1, 1, 1, 1, 0]), 3)
    3

    # n = 9, k = 3, and 8 true bits
    >>> plateau(np.array([1, 1, 0, 1, 1, 1, 1, 1, 1]), 3)
    3

    # n = 9, k = 3, and 9 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]), 3)
    0

    # n = 10, k = 2, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 2)
    10

    # n = 10, k = 2, and 1 true bit
    >>> plateau(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), 2)
    9

    # n = 10, k = 2, and 2 true bits
    >>> plateau(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 2)
    8

    # n = 10, k = 2, and 3 true bits
    >>> plateau(np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 1]), 2)
    7

    # n = 10, k = 2, and 4 true bits
    >>> plateau(np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 1]), 2)
    6

    # n = 10, k = 2, and 5 true bits
    >>> plateau(np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0]), 2)
    5

    # n = 10, k = 2, and 6 true bits
    >>> plateau(np.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 1]), 2)
    4

    # n = 10, k = 2, and 7 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 0, 1, 0, 0, 1]), 2)
    3

    # n = 10, k = 2, and 8 true bits
    >>> plateau(np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]), 2)
    2

    # n = 10, k = 2, and 9 true bits
    >>> plateau(np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1]), 2)
    2

    # n = 10, k = 2, and 10 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 10, k = 3, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 3)
    10

    # n = 10, k = 3, and 1 true bit
    >>> plateau(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), 3)
    9

    # n = 10, k = 3, and 2 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0]), 3)
    8

    # n = 10, k = 3, and 3 true bits
    >>> plateau(np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0]), 3)
    7

    # n = 10, k = 3, and 4 true bits
    >>> plateau(np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0]), 3)
    6

    # n = 10, k = 3, and 5 true bits
    >>> plateau(np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0]), 3)
    5

    # n = 10, k = 3, and 6 true bits
    >>> plateau(np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1]), 3)
    4

    # n = 10, k = 3, and 7 true bits
    >>> plateau(np.array([0, 1, 0, 1, 1, 1, 1, 0, 1, 1]), 3)
    3

    # n = 10, k = 3, and 8 true bits
    >>> plateau(np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1]), 3)
    3

    # n = 10, k = 3, and 9 true bits
    >>> plateau(np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1]), 3)
    3

    # n = 10, k = 3, and 10 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 3)
    0

    # n = 10, k = 4, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4)
    10

    # n = 10, k = 4, and 1 true bit
    >>> plateau(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 4)
    9

    # n = 10, k = 4, and 2 true bits
    >>> plateau(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), 4)
    8

    # n = 10, k = 4, and 3 true bits
    >>> plateau(np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0]), 4)
    7

    # n = 10, k = 4, and 4 true bits
    >>> plateau(np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1]), 4)
    6

    # n = 10, k = 4, and 5 true bits
    >>> plateau(np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0]), 4)
    5

    # n = 10, k = 4, and 6 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 0]), 4)
    4

    # n = 10, k = 4, and 7 true bits
    >>> plateau(np.array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0]), 4)
    4

    # n = 10, k = 4, and 8 true bits
    >>> plateau(np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]), 4)
    4

    # n = 10, k = 4, and 9 true bits
    >>> plateau(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 4)
    4

    # n = 10, k = 4, and 10 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 4)
    0

    # n = 11, k = 2, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 2)
    11

    # n = 11, k = 2, and 1 true bit
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 2)
    10

    # n = 11, k = 2, and 2 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]), 2)
    9

    # n = 11, k = 2, and 3 true bits
    >>> plateau(np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]), 2)
    8

    # n = 11, k = 2, and 4 true bits
    >>> plateau(np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]), 2)
    7

    # n = 11, k = 2, and 5 true bits
    >>> plateau(np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]), 2)
    6

    # n = 11, k = 2, and 6 true bits
    >>> plateau(np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0]), 2)
    5

    # n = 11, k = 2, and 7 true bits
    >>> plateau(np.array([0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]), 2)
    4

    # n = 11, k = 2, and 8 true bits
    >>> plateau(np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]), 2)
    3

    # n = 11, k = 2, and 9 true bits
    >>> plateau(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]), 2)
    2

    # n = 11, k = 2, and 10 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]), 2)
    2

    # n = 11, k = 2, and 11 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 11, k = 3, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 3)
    11

    # n = 11, k = 3, and 1 true bit
    >>> plateau(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), 3)
    10

    # n = 11, k = 3, and 2 true bits
    >>> plateau(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]), 3)
    9

    # n = 11, k = 3, and 3 true bits
    >>> plateau(np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]), 3)
    8

    # n = 11, k = 3, and 4 true bits
    >>> plateau(np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1]), 3)
    7

    # n = 11, k = 3, and 5 true bits
    >>> plateau(np.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]), 3)
    6

    # n = 11, k = 3, and 6 true bits
    >>> plateau(np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]), 3)
    5

    # n = 11, k = 3, and 7 true bits
    >>> plateau(np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0]), 3)
    4

    # n = 11, k = 3, and 8 true bits
    >>> plateau(np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0]), 3)
    3

    # n = 11, k = 3, and 9 true bits
    >>> plateau(np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 3)
    3

    # n = 11, k = 3, and 10 true bits
    >>> plateau(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 3)
    3

    # n = 11, k = 3, and 11 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 3)
    0

    # n = 11, k = 4, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4)
    11

    # n = 11, k = 4, and 1 true bit
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 4)
    10

    # n = 11, k = 4, and 2 true bits
    >>> plateau(np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]), 4)
    9

    # n = 11, k = 4, and 3 true bits
    >>> plateau(np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]), 4)
    8

    # n = 11, k = 4, and 4 true bits
    >>> plateau(np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]), 4)
    7

    # n = 11, k = 4, and 5 true bits
    >>> plateau(np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1]), 4)
    6

    # n = 11, k = 4, and 6 true bits
    >>> plateau(np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0]), 4)
    5

    # n = 11, k = 4, and 7 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]), 4)
    4

    # n = 11, k = 4, and 8 true bits
    >>> plateau(np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1]), 4)
    4

    # n = 11, k = 4, and 9 true bits
    >>> plateau(np.array([1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]), 4)
    4

    # n = 11, k = 4, and 10 true bits
    >>> plateau(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 4)
    4

    # n = 11, k = 4, and 11 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 4)
    0

    # n = 12, k = 2, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 2)
    12

    # n = 12, k = 2, and 1 true bit
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 2)
    11

    # n = 12, k = 2, and 2 true bits
    >>> plateau(np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 2)
    10

    # n = 12, k = 2, and 3 true bits
    >>> plateau(np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]), 2)
    9

    # n = 12, k = 2, and 4 true bits
    >>> plateau(np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]), 2)
    8

    # n = 12, k = 2, and 5 true bits
    >>> plateau(np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]), 2)
    7

    # n = 12, k = 2, and 6 true bits
    >>> plateau(np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1]), 2)
    6

    # n = 12, k = 2, and 7 true bits
    >>> plateau(np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1]), 2)
    5

    # n = 12, k = 2, and 8 true bits
    >>> plateau(np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1]), 2)
    4

    # n = 12, k = 2, and 9 true bits
    >>> plateau(np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]), 2)
    3

    # n = 12, k = 2, and 10 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]), 2)
    2

    # n = 12, k = 2, and 11 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]), 2)
    2

    # n = 12, k = 2, and 12 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 12, k = 3, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 3)
    12

    # n = 12, k = 3, and 1 true bit
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), 3)
    11

    # n = 12, k = 3, and 2 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]), 3)
    10

    # n = 12, k = 3, and 3 true bits
    >>> plateau(np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]), 3)
    9

    # n = 12, k = 3, and 4 true bits
    >>> plateau(np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]), 3)
    8

    # n = 12, k = 3, and 5 true bits
    >>> plateau(np.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1]), 3)
    7

    # n = 12, k = 3, and 6 true bits
    >>> plateau(np.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1]), 3)
    6

    # n = 12, k = 3, and 7 true bits
    >>> plateau(np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]), 3)
    5

    # n = 12, k = 3, and 8 true bits
    >>> plateau(np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0]), 3)
    4

    # n = 12, k = 3, and 9 true bits
    >>> plateau(np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]), 3)
    3

    # n = 12, k = 3, and 10 true bits
    >>> plateau(np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]), 3)
    3

    # n = 12, k = 3, and 11 true bits
    >>> plateau(np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 3)
    3

    # n = 12, k = 3, and 12 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 3)
    0

    # n = 12, k = 4, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4)
    12

    # n = 12, k = 4, and 1 true bit
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 4)
    11

    # n = 12, k = 4, and 2 true bits
    >>> plateau(np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 4)
    10

    # n = 12, k = 4, and 3 true bits
    >>> plateau(np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]), 4)
    9

    # n = 12, k = 4, and 4 true bits
    >>> plateau(np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0]), 4)
    8

    # n = 12, k = 4, and 5 true bits
    >>> plateau(np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]), 4)
    7

    # n = 12, k = 4, and 6 true bits
    >>> plateau(np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0]), 4)
    6

    # n = 12, k = 4, and 7 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0]), 4)
    5

    # n = 12, k = 4, and 8 true bits
    >>> plateau(np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]), 4)
    4

    # n = 12, k = 4, and 9 true bits
    >>> plateau(np.array([1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]), 4)
    4

    # n = 12, k = 4, and 10 true bits
    >>> plateau(np.array([1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]), 4)
    4

    # n = 12, k = 4, and 11 true bits
    >>> plateau(np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]), 4)
    4

    # n = 12, k = 4, and 12 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 4)
    0

    # n = 12, k = 5, and 0 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 5)
    12

    # n = 12, k = 5, and 1 true bit
    >>> plateau(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 5)
    11

    # n = 12, k = 5, and 2 true bits
    >>> plateau(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]), 5)
    10

    # n = 12, k = 5, and 3 true bits
    >>> plateau(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]), 5)
    9

    # n = 12, k = 5, and 4 true bits
    >>> plateau(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]), 5)
    8

    # n = 12, k = 5, and 5 true bits
    >>> plateau(np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]), 5)
    7

    # n = 12, k = 5, and 6 true bits
    >>> plateau(np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0]), 5)
    6

    # n = 12, k = 5, and 7 true bits
    >>> plateau(np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]), 5)
    5

    # n = 12, k = 5, and 8 true bits
    >>> plateau(np.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0]), 5)
    5

    # n = 12, k = 5, and 9 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]), 5)
    5

    # n = 12, k = 5, and 10 true bits
    >>> plateau(np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1]), 5)
    5

    # n = 12, k = 5, and 11 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]), 5)
    5

    # n = 12, k = 5, and 12 true bits
    >>> plateau(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 5)
    0
    """
    res: Final[int] = int(x.sum())
    n: Final[int] = len(x)
    return 0 if res >= n else n - res if res <= (n - k) else k


class Plateau(BitStringNKProblem):
    """Compute the Plateau problem."""

    def __str__(self) -> str:
        """
        Get the name of the plateau objective function.

        :return: `plateau_` + length of string + `_` + k

        >>> Plateau(13, 4)
        plateau_13_4
        """
        return f"plateau_{self.n}_{self.k}"

    def evaluate(self, x: np.ndarray) -> int:
        """
        Evaluate a solution to the plateau problem.

        :param x: the bit string to evaluate
        :returns: the value of the plateau problem for the string
        """
        return plateau(x, self.k)
