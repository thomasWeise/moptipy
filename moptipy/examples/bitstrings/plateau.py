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
   Efficient. *IEEE Transactions on Evolutionary Computation (TEVC)*.
   27(4):980-992. August 2023.
   doi: https://doi.org/10.1109/TEVC.2022.3191698

This is code is part of the research work of Mr. Jiazheng ZENG (曾嘉政),
a Master's student at the Institute of Applied Optimization
(应用优化研究所) of the School of Artificial
Intelligence and Big Data (人工智能与大数据学院) at
Hefei University (合肥大学) in
Hefei, Anhui, China (中国安徽省合肥市) under the supervision of
Prof. Dr. Thomas Weise (汤卫思教授).

>>> len(list(Plateau.default_instances()))
122

>>> [x() for x in Plateau.default_instances()]
[plateau_6_2, plateau_7_2, plateau_8_2, plateau_8_3, plateau_9_2, \
plateau_9_3, plateau_10_2, plateau_10_3, plateau_10_4, plateau_11_2, \
plateau_11_3, plateau_11_4, plateau_12_2, plateau_12_3, plateau_12_4, \
plateau_12_5, plateau_13_2, plateau_13_3, plateau_13_4, plateau_13_5, \
plateau_14_2, plateau_14_3, plateau_14_4, plateau_14_6, plateau_15_2, \
plateau_15_3, plateau_15_4, plateau_15_6, plateau_16_2, plateau_16_4, \
plateau_16_5, plateau_16_7, plateau_17_2, plateau_17_4, plateau_17_5, \
plateau_17_7, plateau_18_2, plateau_18_4, plateau_18_6, plateau_18_8, \
plateau_19_2, plateau_19_4, plateau_19_6, plateau_19_8, plateau_20_2, \
plateau_20_3, plateau_20_4, plateau_20_5, plateau_20_7, plateau_20_9, \
plateau_21_2, plateau_21_3, plateau_21_4, plateau_21_5, plateau_21_7, \
plateau_21_9, plateau_22_2, plateau_22_3, plateau_22_4, plateau_22_5, \
plateau_22_7, plateau_22_10, plateau_23_2, plateau_23_3, plateau_23_4, \
plateau_23_5, plateau_23_7, plateau_23_10, plateau_24_2, plateau_24_3, \
plateau_24_4, plateau_24_6, plateau_24_8, plateau_24_11, plateau_25_2, \
plateau_25_3, plateau_25_5, plateau_25_6, plateau_25_8, plateau_25_11, \
plateau_26_2, plateau_26_3, plateau_26_5, plateau_26_6, plateau_26_9, \
plateau_26_12, plateau_27_2, plateau_27_3, plateau_27_5, plateau_27_6, \
plateau_27_9, plateau_27_12, plateau_28_2, plateau_28_4, plateau_28_5, \
plateau_28_7, plateau_28_10, plateau_28_13, plateau_29_2, plateau_29_4, \
plateau_29_5, plateau_29_7, plateau_29_10, plateau_29_13, plateau_30_2, \
plateau_30_4, plateau_30_5, plateau_30_7, plateau_30_10, plateau_30_14, \
plateau_31_2, plateau_31_4, plateau_31_5, plateau_31_7, plateau_31_10, \
plateau_31_14, plateau_32_2, plateau_32_4, plateau_32_5, plateau_32_8, \
plateau_32_11, plateau_32_15]
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
