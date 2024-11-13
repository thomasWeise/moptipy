"""
The Jump problem.

The Jump problem is basically OneMax, but with a deceptive region of `k` bit
flips before the optimum.
The optimal objective value is 0, which is reached if all bits are `True`.
The worst objective value is `n + k - 1`, which is reached if exactly one
bit is `False`.

1. Stefan Droste, Thomas Jansen, and Ingo Wegener. On the Analysis of the
   (1+1) Evolutionary Algorithm. *Theoretical Computer Science.*
   276(1-2):51-81. April 2002.
   doi: https://doi.org/10.1016/S0304-3975(01)00182-7
2. Tobias Friedrich, Francesco Quinzan, and Markus Wagner. Escaping Large
   Deceptive Basins of Attraction with Heavy-Tailed Mutation Operators. In
   Hernán E. Aguirre and Keiki Takadama, editors, *Proceedings of the Genetic
   and Evolutionary Computation Conference (GECCO'18),* July 15-19, 2018.
   Kyoto, Japan. ACM. doi: https://doi.org/10.1145/3205455.3205515}
3. Francesco Quinzan, Andreas Göbel, Markus Wagner, and Tobias Friedrich.
   Evolutionary algorithms and submodular functions: benefits of heavy-tailed
   mutations. *Natural Computing.* 20(3):561-575. September 2021.
   doi: https://doi.org/10.1007/s11047-021-09841-7.
   Also: arXiv:1805.10902v2 [cs.DS] 21 Nov 2018.
   https://arxiv.org/abs/1805.10902
4. Thomas Weise, Zhize Wu, Xinlu Li, and Yan Chen. Frequency Fitness
   Assignment: Making Optimization Algorithms Invariant under Bijective
   Transformations of the Objective Function Value. *IEEE Transactions on
   Evolutionary Computation* 25(2):307-319. April 2021. Preprint available at
   arXiv:2001.01416v5 [cs.NE] 15 Oct 2020.
   https://dx.doi.org/10.1109/TEVC.2020.3032090
5. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
   Fitness Assignment: Optimization without Bias for Good Solutions can be
   Efficient. *IEEE Transactions on Evolutionary Computation (TEVC)*.
   27(4):980-992. August 2023.
   doi: https://doi.org/10.1109/TEVC.2022.3191698

>>> len(list(Jump.default_instances()))
122

>>> [x() for x in Jump.default_instances()]
[jump_6_2, jump_7_2, jump_8_2, jump_8_3, jump_9_2, jump_9_3, jump_10_2, \
jump_10_3, jump_10_4, jump_11_2, jump_11_3, jump_11_4, jump_12_2, jump_12_3, \
jump_12_4, jump_12_5, jump_13_2, jump_13_3, jump_13_4, jump_13_5, jump_14_2, \
jump_14_3, jump_14_4, jump_14_6, jump_15_2, jump_15_3, jump_15_4, jump_15_6, \
jump_16_2, jump_16_4, jump_16_5, jump_16_7, jump_17_2, jump_17_4, jump_17_5, \
jump_17_7, jump_18_2, jump_18_4, jump_18_6, jump_18_8, jump_19_2, jump_19_4, \
jump_19_6, jump_19_8, jump_20_2, jump_20_3, jump_20_4, jump_20_5, jump_20_7, \
jump_20_9, jump_21_2, jump_21_3, jump_21_4, jump_21_5, jump_21_7, jump_21_9, \
jump_22_2, jump_22_3, jump_22_4, jump_22_5, jump_22_7, jump_22_10, \
jump_23_2, jump_23_3, jump_23_4, jump_23_5, jump_23_7, jump_23_10, \
jump_24_2, jump_24_3, jump_24_4, jump_24_6, jump_24_8, jump_24_11, \
jump_25_2, jump_25_3, jump_25_5, jump_25_6, jump_25_8, jump_25_11, \
jump_26_2, jump_26_3, jump_26_5, jump_26_6, jump_26_9, jump_26_12, \
jump_27_2, jump_27_3, jump_27_5, jump_27_6, jump_27_9, jump_27_12, \
jump_28_2, jump_28_4, jump_28_5, jump_28_7, jump_28_10, jump_28_13, \
jump_29_2, jump_29_4, jump_29_5, jump_29_7, jump_29_10, jump_29_13, \
jump_30_2, jump_30_4, jump_30_5, jump_30_7, jump_30_10, jump_30_14, \
jump_31_2, jump_31_4, jump_31_5, jump_31_7, jump_31_10, jump_31_14, \
jump_32_2, jump_32_4, jump_32_5, jump_32_8, jump_32_11, jump_32_15]
"""

from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.examples.bitstrings.bitstring_problem import BitStringNKProblem


@numba.njit(nogil=True, cache=True)
def jump(x: np.ndarray, k: int) -> int:
    """
    Compute the jump value.

    :param x: the np array
    :param k: the k parameter
    :return: jump value

    >>> jump(np.array([False, False, False, False, False, False]), 2)
    6
    >>> jump(np.array([False, False, False, False, True, False]), 2)
    5
    >>> jump(np.array([False, True, True, False, False, False]), 2)
    4
    >>> jump(np.array([True, False, True, False, True, False]), 2)
    3
    >>> jump(np.array([True, False, True, False, True, True]), 2)
    2
    >>> jump(np.array([True, True, True, True, True, False]), 2)
    7
    >>> jump(np.array([True, True, True, True, True, True]), 2)
    0

    # n = 6, k = 2, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0]), 2)
    6

    # n = 6, k = 2, and 1 true bit
    >>> jump(np.array([0, 0, 0, 1, 0, 0]), 2)
    5

    # n = 6, k = 2, and 2 true bits
    >>> jump(np.array([0, 0, 1, 0, 0, 1]), 2)
    4

    # n = 6, k = 2, and 3 true bits
    >>> jump(np.array([1, 1, 1, 0, 0, 0]), 2)
    3

    # n = 6, k = 2, and 4 true bits
    >>> jump(np.array([1, 0, 0, 1, 1, 1]), 2)
    2

    # n = 6, k = 2, and 5 true bits
    >>> jump(np.array([1, 1, 1, 1, 0, 1]), 2)
    7

    # n = 6, k = 2, and 6 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 7, k = 2, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0]), 2)
    7

    # n = 7, k = 2, and 1 true bit
    >>> jump(np.array([0, 0, 0, 1, 0, 0, 0]), 2)
    6

    # n = 7, k = 2, and 2 true bits
    >>> jump(np.array([0, 0, 1, 0, 1, 0, 0]), 2)
    5

    # n = 7, k = 2, and 3 true bits
    >>> jump(np.array([1, 1, 0, 1, 0, 0, 0]), 2)
    4

    # n = 7, k = 2, and 4 true bits
    >>> jump(np.array([0, 1, 0, 1, 1, 1, 0]), 2)
    3

    # n = 7, k = 2, and 5 true bits
    >>> jump(np.array([1, 0, 1, 1, 0, 1, 1]), 2)
    2

    # n = 7, k = 2, and 6 true bits
    >>> jump(np.array([1, 1, 1, 1, 0, 1, 1]), 2)
    8

    # n = 7, k = 2, and 7 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 8, k = 2, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0]), 2)
    8

    # n = 8, k = 2, and 1 true bit
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 1, 0]), 2)
    7

    # n = 8, k = 2, and 2 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 1, 1]), 2)
    6

    # n = 8, k = 2, and 3 true bits
    >>> jump(np.array([0, 0, 1, 0, 0, 0, 1, 1]), 2)
    5

    # n = 8, k = 2, and 4 true bits
    >>> jump(np.array([0, 0, 0, 1, 1, 1, 0, 1]), 2)
    4

    # n = 8, k = 2, and 5 true bits
    >>> jump(np.array([0, 1, 1, 1, 1, 1, 0, 0]), 2)
    3

    # n = 8, k = 2, and 6 true bits
    >>> jump(np.array([1, 1, 1, 0, 1, 1, 1, 0]), 2)
    2

    # n = 8, k = 2, and 7 true bits
    >>> jump(np.array([1, 1, 1, 1, 0, 1, 1, 1]), 2)
    9

    # n = 8, k = 2, and 8 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 8, k = 3, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0]), 3)
    8

    # n = 8, k = 3, and 1 true bit
    >>> jump(np.array([0, 1, 0, 0, 0, 0, 0, 0]), 3)
    7

    # n = 8, k = 3, and 2 true bits
    >>> jump(np.array([0, 0, 0, 0, 1, 0, 0, 1]), 3)
    6

    # n = 8, k = 3, and 3 true bits
    >>> jump(np.array([0, 0, 0, 1, 1, 1, 0, 0]), 3)
    5

    # n = 8, k = 3, and 4 true bits
    >>> jump(np.array([0, 1, 0, 0, 1, 1, 1, 0]), 3)
    4

    # n = 8, k = 3, and 5 true bits
    >>> jump(np.array([1, 0, 1, 1, 1, 0, 0, 1]), 3)
    3

    # n = 8, k = 3, and 6 true bits
    >>> jump(np.array([1, 0, 1, 0, 1, 1, 1, 1]), 3)
    9

    # n = 8, k = 3, and 7 true bits
    >>> jump(np.array([1, 0, 1, 1, 1, 1, 1, 1]), 3)
    10

    # n = 8, k = 3, and 8 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1]), 3)
    0

    # n = 9, k = 2, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), 2)
    9

    # n = 9, k = 2, and 1 true bit
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]), 2)
    8

    # n = 9, k = 2, and 2 true bits
    >>> jump(np.array([0, 0, 1, 0, 0, 0, 0, 0, 1]), 2)
    7

    # n = 9, k = 2, and 3 true bits
    >>> jump(np.array([0, 0, 0, 1, 0, 0, 1, 0, 1]), 2)
    6

    # n = 9, k = 2, and 4 true bits
    >>> jump(np.array([1, 0, 0, 0, 1, 0, 0, 1, 1]), 2)
    5

    # n = 9, k = 2, and 5 true bits
    >>> jump(np.array([0, 0, 0, 1, 1, 0, 1, 1, 1]), 2)
    4

    # n = 9, k = 2, and 6 true bits
    >>> jump(np.array([1, 1, 0, 1, 0, 1, 1, 1, 0]), 2)
    3

    # n = 9, k = 2, and 7 true bits
    >>> jump(np.array([1, 0, 1, 1, 1, 1, 0, 1, 1]), 2)
    2

    # n = 9, k = 2, and 8 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 0, 1, 1, 1]), 2)
    10

    # n = 9, k = 2, and 9 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 9, k = 3, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), 3)
    9

    # n = 9, k = 3, and 1 true bit
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]), 3)
    8

    # n = 9, k = 3, and 2 true bits
    >>> jump(np.array([0, 0, 1, 0, 1, 0, 0, 0, 0]), 3)
    7

    # n = 9, k = 3, and 3 true bits
    >>> jump(np.array([0, 1, 0, 1, 1, 0, 0, 0, 0]), 3)
    6

    # n = 9, k = 3, and 4 true bits
    >>> jump(np.array([0, 1, 1, 0, 1, 0, 0, 1, 0]), 3)
    5

    # n = 9, k = 3, and 5 true bits
    >>> jump(np.array([1, 1, 1, 1, 0, 0, 0, 0, 1]), 3)
    4

    # n = 9, k = 3, and 6 true bits
    >>> jump(np.array([0, 0, 1, 1, 1, 1, 1, 0, 1]), 3)
    3

    # n = 9, k = 3, and 7 true bits
    >>> jump(np.array([0, 1, 1, 1, 1, 1, 1, 1, 0]), 3)
    10

    # n = 9, k = 3, and 8 true bits
    >>> jump(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1]), 3)
    11

    # n = 9, k = 3, and 9 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]), 3)
    0

    # n = 10, k = 2, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 2)
    10

    # n = 10, k = 2, and 1 true bit
    >>> jump(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]), 2)
    9

    # n = 10, k = 2, and 2 true bits
    >>> jump(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1]), 2)
    8

    # n = 10, k = 2, and 3 true bits
    >>> jump(np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1]), 2)
    7

    # n = 10, k = 2, and 4 true bits
    >>> jump(np.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 1]), 2)
    6

    # n = 10, k = 2, and 5 true bits
    >>> jump(np.array([1, 0, 1, 1, 0, 1, 1, 0, 0, 0]), 2)
    5

    # n = 10, k = 2, and 6 true bits
    >>> jump(np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1]), 2)
    4

    # n = 10, k = 2, and 7 true bits
    >>> jump(np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 1]), 2)
    3

    # n = 10, k = 2, and 8 true bits
    >>> jump(np.array([1, 0, 1, 1, 1, 1, 1, 0, 1, 1]), 2)
    2

    # n = 10, k = 2, and 9 true bits
    >>> jump(np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1]), 2)
    11

    # n = 10, k = 2, and 10 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 10, k = 3, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 3)
    10

    # n = 10, k = 3, and 1 true bit
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 3)
    9

    # n = 10, k = 3, and 2 true bits
    >>> jump(np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 3)
    8

    # n = 10, k = 3, and 3 true bits
    >>> jump(np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0]), 3)
    7

    # n = 10, k = 3, and 4 true bits
    >>> jump(np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0]), 3)
    6

    # n = 10, k = 3, and 5 true bits
    >>> jump(np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1]), 3)
    5

    # n = 10, k = 3, and 6 true bits
    >>> jump(np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 1]), 3)
    4

    # n = 10, k = 3, and 7 true bits
    >>> jump(np.array([0, 1, 1, 1, 1, 1, 0, 0, 1, 1]), 3)
    3

    # n = 10, k = 3, and 8 true bits
    >>> jump(np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1]), 3)
    11

    # n = 10, k = 3, and 9 true bits
    >>> jump(np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1]), 3)
    12

    # n = 10, k = 3, and 10 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 3)
    0

    # n = 10, k = 4, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4)
    10

    # n = 10, k = 4, and 1 true bit
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 4)
    9

    # n = 10, k = 4, and 2 true bits
    >>> jump(np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0]), 4)
    8

    # n = 10, k = 4, and 3 true bits
    >>> jump(np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 4)
    7

    # n = 10, k = 4, and 4 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1]), 4)
    6

    # n = 10, k = 4, and 5 true bits
    >>> jump(np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0]), 4)
    5

    # n = 10, k = 4, and 6 true bits
    >>> jump(np.array([1, 1, 1, 1, 0, 1, 0, 0, 1, 0]), 4)
    4

    # n = 10, k = 4, and 7 true bits
    >>> jump(np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 0]), 4)
    11

    # n = 10, k = 4, and 8 true bits
    >>> jump(np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 1]), 4)
    12

    # n = 10, k = 4, and 9 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1]), 4)
    13

    # n = 10, k = 4, and 10 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 4)
    0

    # n = 11, k = 2, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 2)
    11

    # n = 11, k = 2, and 1 true bit
    >>> jump(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 2)
    10

    # n = 11, k = 2, and 2 true bits
    >>> jump(np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]), 2)
    9

    # n = 11, k = 2, and 3 true bits
    >>> jump(np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0]), 2)
    8

    # n = 11, k = 2, and 4 true bits
    >>> jump(np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0]), 2)
    7

    # n = 11, k = 2, and 5 true bits
    >>> jump(np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1]), 2)
    6

    # n = 11, k = 2, and 6 true bits
    >>> jump(np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]), 2)
    5

    # n = 11, k = 2, and 7 true bits
    >>> jump(np.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]), 2)
    4

    # n = 11, k = 2, and 8 true bits
    >>> jump(np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0]), 2)
    3

    # n = 11, k = 2, and 9 true bits
    >>> jump(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]), 2)
    2

    # n = 11, k = 2, and 10 true bits
    >>> jump(np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]), 2)
    12

    # n = 11, k = 2, and 11 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 11, k = 3, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 3)
    11

    # n = 11, k = 3, and 1 true bit
    >>> jump(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), 3)
    10

    # n = 11, k = 3, and 2 true bits
    >>> jump(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 3)
    9

    # n = 11, k = 3, and 3 true bits
    >>> jump(np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]), 3)
    8

    # n = 11, k = 3, and 4 true bits
    >>> jump(np.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]), 3)
    7

    # n = 11, k = 3, and 5 true bits
    >>> jump(np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0]), 3)
    6

    # n = 11, k = 3, and 6 true bits
    >>> jump(np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1]), 3)
    5

    # n = 11, k = 3, and 7 true bits
    >>> jump(np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0]), 3)
    4

    # n = 11, k = 3, and 8 true bits
    >>> jump(np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]), 3)
    3

    # n = 11, k = 3, and 9 true bits
    >>> jump(np.array([1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]), 3)
    12

    # n = 11, k = 3, and 10 true bits
    >>> jump(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 3)
    13

    # n = 11, k = 3, and 11 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 3)
    0

    # n = 11, k = 4, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4)
    11

    # n = 11, k = 4, and 1 true bit
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 4)
    10

    # n = 11, k = 4, and 2 true bits
    >>> jump(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 4)
    9

    # n = 11, k = 4, and 3 true bits
    >>> jump(np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]), 4)
    8

    # n = 11, k = 4, and 4 true bits
    >>> jump(np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]), 4)
    7

    # n = 11, k = 4, and 5 true bits
    >>> jump(np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0]), 4)
    6

    # n = 11, k = 4, and 6 true bits
    >>> jump(np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1]), 4)
    5

    # n = 11, k = 4, and 7 true bits
    >>> jump(np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]), 4)
    4

    # n = 11, k = 4, and 8 true bits
    >>> jump(np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]), 4)
    12

    # n = 11, k = 4, and 9 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), 4)
    13

    # n = 11, k = 4, and 10 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]), 4)
    14

    # n = 11, k = 4, and 11 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 4)
    0

    # n = 12, k = 2, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 2)
    12

    # n = 12, k = 2, and 1 true bit
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 2)
    11

    # n = 12, k = 2, and 2 true bits
    >>> jump(np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 2)
    10

    # n = 12, k = 2, and 3 true bits
    >>> jump(np.array([0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]), 2)
    9

    # n = 12, k = 2, and 4 true bits
    >>> jump(np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 2)
    8

    # n = 12, k = 2, and 5 true bits
    >>> jump(np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]), 2)
    7

    # n = 12, k = 2, and 6 true bits
    >>> jump(np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0]), 2)
    6

    # n = 12, k = 2, and 7 true bits
    >>> jump(np.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0]), 2)
    5

    # n = 12, k = 2, and 8 true bits
    >>> jump(np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1]), 2)
    4

    # n = 12, k = 2, and 9 true bits
    >>> jump(np.array([1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]), 2)
    3

    # n = 12, k = 2, and 10 true bits
    >>> jump(np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]), 2)
    2

    # n = 12, k = 2, and 11 true bits
    >>> jump(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 2)
    13

    # n = 12, k = 2, and 12 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 2)
    0

    # n = 12, k = 3, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 3)
    12

    # n = 12, k = 3, and 1 true bit
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 3)
    11

    # n = 12, k = 3, and 2 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]), 3)
    10

    # n = 12, k = 3, and 3 true bits
    >>> jump(np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]), 3)
    9

    # n = 12, k = 3, and 4 true bits
    >>> jump(np.array([0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0]), 3)
    8

    # n = 12, k = 3, and 5 true bits
    >>> jump(np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0]), 3)
    7

    # n = 12, k = 3, and 6 true bits
    >>> jump(np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]), 3)
    6

    # n = 12, k = 3, and 7 true bits
    >>> jump(np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]), 3)
    5

    # n = 12, k = 3, and 8 true bits
    >>> jump(np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]), 3)
    4

    # n = 12, k = 3, and 9 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0]), 3)
    3

    # n = 12, k = 3, and 10 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1]), 3)
    13

    # n = 12, k = 3, and 11 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]), 3)
    14

    # n = 12, k = 3, and 12 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 3)
    0

    # n = 12, k = 4, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4)
    12

    # n = 12, k = 4, and 1 true bit
    >>> jump(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), 4)
    11

    # n = 12, k = 4, and 2 true bits
    >>> jump(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), 4)
    10

    # n = 12, k = 4, and 3 true bits
    >>> jump(np.array([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]), 4)
    9

    # n = 12, k = 4, and 4 true bits
    >>> jump(np.array([1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]), 4)
    8

    # n = 12, k = 4, and 5 true bits
    >>> jump(np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0]), 4)
    7

    # n = 12, k = 4, and 6 true bits
    >>> jump(np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]), 4)
    6

    # n = 12, k = 4, and 7 true bits
    >>> jump(np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1]), 4)
    5

    # n = 12, k = 4, and 8 true bits
    >>> jump(np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]), 4)
    4

    # n = 12, k = 4, and 9 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]), 4)
    13

    # n = 12, k = 4, and 10 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]), 4)
    14

    # n = 12, k = 4, and 11 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]), 4)
    15

    # n = 12, k = 4, and 12 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 4)
    0

    # n = 12, k = 5, and 0 true bits
    >>> jump(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 5)
    12

    # n = 12, k = 5, and 1 true bit
    >>> jump(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 5)
    11

    # n = 12, k = 5, and 2 true bits
    >>> jump(np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 5)
    10

    # n = 12, k = 5, and 3 true bits
    >>> jump(np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]), 5)
    9

    # n = 12, k = 5, and 4 true bits
    >>> jump(np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1]), 5)
    8

    # n = 12, k = 5, and 5 true bits
    >>> jump(np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0]), 5)
    7

    # n = 12, k = 5, and 6 true bits
    >>> jump(np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0]), 5)
    6

    # n = 12, k = 5, and 7 true bits
    >>> jump(np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]), 5)
    5

    # n = 12, k = 5, and 8 true bits
    >>> jump(np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]), 5)
    13

    # n = 12, k = 5, and 9 true bits
    >>> jump(np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]), 5)
    14

    # n = 12, k = 5, and 10 true bits
    >>> jump(np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]), 5)
    15

    # n = 12, k = 5, and 11 true bits
    >>> jump(np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]), 5)
    16

    # n = 12, k = 5, and 12 true bits
    >>> jump(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 5)
    0
    """
    res: Final[int] = int(x.sum())
    n: Final[int] = len(x)
    nmk: Final[int] = n - k
    return n - res if (res >= n) or (res <= nmk) else k + res


class Jump(BitStringNKProblem):
    """Compute the Jump problem."""

    def __str__(self) -> str:
        """
        Get the name of the jump objective function.

        :return: `jump_` + length of string + `_` + k

        >>> Jump(13, 4)
        jump_13_4
        """
        return f"jump_{self.n}_{self.k}"

    def evaluate(self, x: np.ndarray) -> int:
        """
        Evaluate a solution to the jump problem.

        :param x: the bit string to evaluate
        :returns: the value of the jump problem for the string
        """
        return jump(x, self.k)

    def upper_bound(self) -> int:
        """
        Get the upper bound of the jump problem.

        :return: the length of the bit string + the length of the jump - 1

        >>> Jump(15, 4).upper_bound()
        18
        """
        return self.n + self.k - 1
