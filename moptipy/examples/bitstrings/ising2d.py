"""
The two-dimensional Ising problem.

The two-dimensional Ising problem describes a torus. For each bit that differs
from its right-side neighboring bit, a penalty of 1 is incurred. For each bit
differing from its above neighbor, another penalty of 1 is added.
The optimum is a bit string of either all ones or all zeros.
The optimal objective value is 0.
The worst possible objective value is `2*n`.

1. Simon Fischer. A Polynomial Upper Bound for a Mutation-Based Algorithm on
   the Two-Dimensional Ising Model. In Kalyanmoy Deb, editor, Genetic and
   Evolutionary Computation Conference, June 26-30, 2004, Seattle, WA, USA,
   Proceedings, Part I, Lecture Notes in Computer Science (LNCS), volume 3102,
   Springer Berlin, Heidelberg. pages 1100-1112.
   doi: https://doi.org/10.1007/978-3-540-24854-5_108.
2. Carola Doerr and Furong Ye and Naama Horesh and Hao Wang and Ofer M. Shir
   and Thomas Bäck. Benchmarking Discrete Optimization Heuristics with
   IOHprofiler. Applied Soft Computing 88:106027, March 2020,
   doi: https://doi.org/10.1016/j.asoc.2019.106027.
3. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
   Fitness Assignment: Optimization without Bias for Good Solutions can be
   Efficient. *IEEE Transactions on Evolutionary Computation (TEVC)*.
   27(4):980-992. August 2023.
   doi: https://doi.org/10.1109/TEVC.2022.3191698

This is code is part of the research work of Mr. Jiazheng ZENG (曾嘉政),
a Master's student at the Institute of Applied Optimization
(应用优化研究所, http://iao.hfuu.edu.cn) of the School of Artificial
Intelligence and Big Data (人工智能与大数据学院) at
Hefei University (合肥大学) in
Hefei, Anhui, China (中国安徽省合肥市) under the supervision of
Prof. Dr. Thomas Weise (汤卫思教授).
"""
from typing import Callable, Final, Iterator, cast

import numba  # type: ignore
import numpy as np

from moptipy.examples.bitstrings.bitstring_problem import (
    SquareBitStringProblem,
)


@numba.njit(nogil=True, cache=True)
def ising2d(x: np.ndarray, k: int) -> int:
    """
    Calculate the two-dimensional Ising model.

    :param x: the flat numpy array representing the bitstring
    :param k: the side length of the square grid (i.e., `sqrt(len(x)))`
    :return: the two-dimensional Ising objective

    >>> test_k = 2
    >>> y = np.array([False] * test_k * test_k)
    >>> ising2d(y, test_k)
    0

    >>> y.fill(True)
    >>> ising2d(y, test_k)
    0

    >>> y[2] = not y[2]
    >>> ising2d(y, test_k)
    4

    >>> ising2d(np.array([True, False, False, True]), 2)
    8

    >>> test_k = 3
    >>> y = np.array([False] * test_k * test_k)
    >>> ising2d(y, test_k)
    0

    >>> y.fill(True)
    >>> ising2d(y, test_k)
    0

    >>> y[4] = not y[4]
    >>> ising2d(y, test_k)
    4

    >>> y[5] = not y[5]
    >>> ising2d(y, test_k)
    6

    >>> y[3] = not y[3]
    >>> ising2d(y, test_k)
    6

    >>> test_k = 5
    >>> y = np.array([False] * test_k * test_k)
    >>> ising2d(y, test_k)
    0

    >>> y.fill(True)
    >>> ising2d(y, test_k)
    0

    >>> y[5] = not y[5]
    >>> ising2d(y, test_k)
    4

    >>> y[6] = not y[6]
    >>> ising2d(y, test_k)
    6

    >>> y[7] = not y[7]
    >>> ising2d(y, test_k)
    8

    >>> y[6] = not y[6]
    >>> ising2d(y, test_k)
    8

    >>> y[10] = not y[10]
    >>> ising2d(y, test_k)
    10

    >>> y[12] = not y[12]
    >>> ising2d(y, test_k)
    12

    >>> y[11] = not y[11]
    >>> ising2d(y, test_k)
    12

    >>> y[6] = not y[6]
    >>> ising2d(y, test_k)
    10

    # 16 bits, torus width = 4, and 2 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), 4)
    8

    # 16 bits, torus width = 4, and 3 true bits
    >>> ising2d(np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), 4)
    12

    # 16 bits, torus width = 4, and 3 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), 4)
    10

    # 16 bits, torus width = 4, and 1 true bit
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 4)
    4

    # 16 bits, torus width = 4, and 4 true bits
    >>> ising2d(np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 4)
    10

    # 16 bits, torus width = 4, and 4 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]), 4)
    10

    # 16 bits, torus width = 4, and 4 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]), 4)
    12

    # 16 bits, torus width = 4, and 4 true bits
    >>> ising2d(np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]), 4)
    12

    # 16 bits, torus width = 4, and 10 true bits
    >>> ising2d(np.array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1]), 4)
    14

    # 16 bits, torus width = 4, and 8 true bits
    >>> ising2d(np.array([1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0]), 4)
    22

    # 16 bits, torus width = 4, and 12 true bits
    >>> ising2d(np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0]), 4)
    12

    # 16 bits, torus width = 4, and 13 true bits
    >>> ising2d(np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]), 4)
    12

    # 16 bits, torus width = 4, and 0 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4)
    0

    # 16 bits, torus width = 4, and 16 true bits
    >>> ising2d(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 4)
    0

    # 25 bits, torus width = 5, and 1 true bit
    >>> ising2d(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0]), 5)
    4

    # 25 bits, torus width = 5, and 1 true bit
    >>> ising2d(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0]), 5)
    4

    # 25 bits, torus width = 5, and 1 true bit
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0]), 5)
    4

    # 25 bits, torus width = 5, and 4 true bits
    >>> ising2d(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    ...    0, 1, 0, 1, 0, 0, 0, 0]), 5)
    12

    # 25 bits, torus width = 5, and 5 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    1, 0, 1, 0, 0, 1, 0, 0]), 5)
    16

    # 25 bits, torus width = 5, and 5 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
    ...    1, 0, 0, 0, 0, 0, 0, 1]), 5)
    16

    # 25 bits, torus width = 5, and 5 true bits
    >>> ising2d(np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 1]), 5)
    18

    # 25 bits, torus width = 5, and 5 true bits
    >>> ising2d(np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 1]), 5)
    20

    # 25 bits, torus width = 5, and 13 true bits
    >>> ising2d(np.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1,
    ...    1, 0, 1, 0, 1, 0, 0, 1]), 5)
    26

    # 25 bits, torus width = 5, and 21 true bits
    >>> ising2d(np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1]), 5)
    12

    # 25 bits, torus width = 5, and 24 true bits
    >>> ising2d(np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1]), 5)
    4

    # 25 bits, torus width = 5, and 12 true bits
    >>> ising2d(np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0,
    ...    0, 1, 0, 0, 1, 0, 1, 0]), 5)
    24

    # 25 bits, torus width = 5, and 0 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0]), 5)
    0

    # 25 bits, torus width = 5, and 25 true bits
    >>> ising2d(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1]), 5)
    0

    # 36 bits, torus width = 6, and 3 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ...    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 6)
    12

    # 36 bits, torus width = 6, and 4 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), 6)
    12

    # 36 bits, torus width = 6, and 2 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 6)
    8

    # 36 bits, torus width = 6, and 1 true bit
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 6)
    4

    # 36 bits, torus width = 6, and 6 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    ...    0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]), 6)
    18

    # 36 bits, torus width = 6, and 6 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0]), 6)
    18

    # 36 bits, torus width = 6, and 6 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]), 6)
    18

    # 36 bits, torus width = 6, and 6 true bits
    >>> ising2d(np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 6)
    18

    # 36 bits, torus width = 6, and 12 true bits
    >>> ising2d(np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0]), 6)
    28

    # 36 bits, torus width = 6, and 15 true bits
    >>> ising2d(np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0,
    ...    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]), 6)
    34

    # 36 bits, torus width = 6, and 26 true bits
    >>> ising2d(np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
    ...    1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1]), 6)
    30

    # 36 bits, torus width = 6, and 20 true bits
    >>> ising2d(np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1,
    ...    1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]), 6)
    32

    # 36 bits, torus width = 6, and 0 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 6)
    0

    # 36 bits, torus width = 6, and 36 true bits
    >>> ising2d(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 6)
    0

    # 49 bits, torus width = 7, and 6 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 0, 0, 0, 1, 0]), 7)
    24

    # 49 bits, torus width = 7, and 4 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 7)
    16

    # 49 bits, torus width = 7, and 2 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 7)
    8

    # 49 bits, torus width = 7, and 1 true bit
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 7)
    4

    # 49 bits, torus width = 7, and 7 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    ...    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
    ...    0, 1, 0, 0, 0, 0, 0, 1, 0, 0]), 7)
    24

    # 49 bits, torus width = 7, and 7 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
    ...    1, 0, 0, 0, 0, 1, 0, 0, 0, 0]), 7)
    26

    # 49 bits, torus width = 7, and 7 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ...    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ...    0, 0, 1, 0, 1, 0, 0, 0, 1, 0]), 7)
    28

    # 49 bits, torus width = 7, and 7 true bits
    >>> ising2d(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ...    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    ...    0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), 7)
    26

    # 49 bits, torus width = 7, and 14 true bits
    >>> ising2d(np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
    ...    0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
    ...    0, 0, 0, 0, 0, 1, 0, 0, 0, 0]), 7)
    36

    # 49 bits, torus width = 7, and 12 true bits
    >>> ising2d(np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    1, 0, 0, 1, 0, 0, 1, 0, 0, 0]), 7)
    38

    # 49 bits, torus width = 7, and 15 true bits
    >>> ising2d(np.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
    ...    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0,
    ...    1, 1, 1, 0, 0, 1, 0, 0, 1, 0]), 7)
    42

    # 49 bits, torus width = 7, and 29 true bits
    >>> ising2d(np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
    ...    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0,
    ...    0, 1, 1, 1, 1, 0, 1, 1, 1, 1]), 7)
    48

    # 49 bits, torus width = 7, and 0 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 7)
    0

    # 49 bits, torus width = 7, and 49 true bits
    >>> ising2d(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 7)
    0

    # 64 bits, torus width = 8, and 5 true bits
    >>> ising2d(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
    ...    0, 0, 0]), 8)
    20

    # 64 bits, torus width = 8, and 7 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
    ...    0, 0, 0]), 8)
    26

    # 64 bits, torus width = 8, and 2 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0]), 8)
    8

    # 64 bits, torus width = 8, and 4 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0]), 8)
    14

    # 64 bits, torus width = 8, and 8 true bits
    >>> ising2d(np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ...    1, 0, 0]), 8)
    30

    # 64 bits, torus width = 8, and 8 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0]), 8)
    28

    # 64 bits, torus width = 8, and 8 true bits
    >>> ising2d(np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
    ...    0, 1, 0]), 8)
    30

    # 64 bits, torus width = 8, and 8 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
    ...    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0]), 8)
    26

    # 64 bits, torus width = 8, and 37 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1,
    ...    1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    ...    0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
    ...    1, 1, 0]), 8)
    58

    # 64 bits, torus width = 8, and 36 true bits
    >>> ising2d(np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1,
    ...    1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    ...    0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1,
    ...    1, 1, 1]), 8)
    62

    # 64 bits, torus width = 8, and 55 true bits
    >>> ising2d(np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
    ...    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
    ...    1, 1, 1]), 8)
    32

    # 64 bits, torus width = 8, and 38 true bits
    >>> ising2d(np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1,
    ...    1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1,
    ...    1, 0, 1]), 8)
    68

    # 64 bits, torus width = 8, and 0 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0]), 8)
    0

    # 64 bits, torus width = 8, and 64 true bits
    >>> ising2d(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1]), 8)
    0

    # 81 bits, torus width = 9, and 1 true bit
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 9)
    4

    # 81 bits, torus width = 9, and 5 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    ...    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 9)
    20

    # 81 bits, torus width = 9, and 1 true bit
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 9)
    4

    # 81 bits, torus width = 9, and 3 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ...    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 9)
    12

    # 81 bits, torus width = 9, and 9 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
    ...    0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
    ...    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), 9)
    30

    # 81 bits, torus width = 9, and 9 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 9)
    32

    # 81 bits, torus width = 9, and 9 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    ...    0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), 9)
    32

    # 81 bits, torus width = 9, and 9 true bits
    >>> ising2d(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]), 9)
    36

    # 81 bits, torus width = 9, and 11 true bits
    >>> ising2d(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
    ...    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 9)
    40

    # 81 bits, torus width = 9, and 40 true bits
    >>> ising2d(np.array([1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0,
    ...    0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1,
    ...    1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1,
    ...    1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0]), 9)
    88

    # 81 bits, torus width = 9, and 35 true bits
    >>> ising2d(np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
    ...    1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0,
    ...    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0,
    ...    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]), 9)
    70

    # 81 bits, torus width = 9, and 79 true bits
    >>> ising2d(np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 9)
    8

    # 81 bits, torus width = 9, and 0 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 9)
    0

    # 81 bits, torus width = 9, and 81 true bits
    >>> ising2d(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 9)
    0

    # 100 bits, torus width = 10, and 3 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), 10)
    12

    # 100 bits, torus width = 10, and 5 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 10)
    20

    # 100 bits, torus width = 10, and 2 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 10)
    8

    # 100 bits, torus width = 10, and 1 true bit
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 10)
    4

    # 100 bits, torus width = 10, and 10 true bits
    >>> ising2d(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    ...    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 10)
    40

    # 100 bits, torus width = 10, and 10 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ...    0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 10)
    38

    # 100 bits, torus width = 10, and 10 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
    ...    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1,
    ...    1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 10)
    28

    # 100 bits, torus width = 10, and 10 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]), 10)
    36

    # 100 bits, torus width = 10, and 91 true bits
    >>> ising2d(np.array([1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    ...    1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 10)
    32

    # 100 bits, torus width = 10, and 32 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
    ...    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    ...    0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0,
    ...    1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    ...    0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]), 10)
    68

    # 100 bits, torus width = 10, and 89 true bits
    >>> ising2d(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]), 10)
    42

    # 100 bits, torus width = 10, and 92 true bits
    >>> ising2d(np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 10)
    30

    # 100 bits, torus width = 10, and 0 true bits
    >>> ising2d(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 10)
    0

    # 100 bits, torus width = 10, and 100 true bits
    >>> ising2d(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 10)
    0
    """
    n: Final[int] = len(x)  # Get the length of the bit string.
    result: int = n + n  # Start with worst-case result.

    prev_row_start: int = n - k  # The start of the bottom row.
    cur_row_start: int = 0  # The start index of the first row.
    for _ in range(k):  # Iterate over the rows.
        prev_j: int = k - 1  # The previous index = last element.
        for j in range(k):  # Iterate over the current row.
            center = x[cur_row_start + j]  # The current element.
            if center == x[prev_row_start + j]:  # One row above the same?
                result -= 1  # If above element same, no penalty.
            if center == x[cur_row_start + prev_j]:  # Element to the right.
                result -= 1  # If right element same, no penalty.
            prev_j = j  # Previous element <-- current element.
        prev_row_start = cur_row_start  # Previous row <-- current row.
        cur_row_start += k  # k elements per row.

    return result


class Ising2d(SquareBitStringProblem):
    """
    The two-dimensional Ising model.

    >>> ising = Ising2d(16)
    >>> ising.n
    16
    >>> ising.k
    4
    >>> ising.evaluate(np.array([False, False, False, False,
    ...                          False, False, False, False,
    ...                          False, False, False, False,
    ...                          False, False, False, False]))
    0
    >>> ising.evaluate(np.array([False, False, False, False,
    ...                          False,  True, False, False,
    ...                          False, False, False, False,
    ...                          False, False, False, False]))
    4
    >>> ising.evaluate(np.array([False, False, False, False,
    ...                           True,  True,  True,  True,
    ...                          False, False, False, False,
    ...                          False, False, False, False]))
    8
    >>> ising.evaluate(np.array([False,  True, False, False,
    ...                           True,  True,  True,  True,
    ...                          False,  True, False, False,
    ...                          False,  True, False, False]))
    12
    >>> ising.evaluate(np.array([False,  True, False, False,
    ...                           True,  True,  True,  True,
    ...                          False,  True, False, False,
    ...                          False,  True, False,  True]))
    16
    >>> ising.evaluate(np.array([False,  True, False, False,
    ...                           True,  True,  True,  True,
    ...                          False,  True, False, False,
    ...                           True,  True,  True,  True]))
    16
    """

    def evaluate(self, x: np.ndarray) -> int:
        """
        Evaluate a solution to the 2D Ising problem.

        :param x: the bit string to evaluate
        :returns: the value of the 2D Ising Model for the string
        """
        return ising2d(x, self.k)

    def upper_bound(self) -> int:
        """
        Get the upper bound of the two-dimensional Ising model.

        :return: twice the length of the bit string

        >>> Ising2d(49).upper_bound()
        98
        """
        return 2 * self.n

    def __str__(self) -> str:
        """
        Get the name of the two-dimensional Ising model objective function.

        :return: ising2d_ + length of string

        >>> Ising2d(16)
        ising2d_16
        """
        return f"ising2d_{self.n}"

    @classmethod
    def default_instances(
            cls: type, scale_min: int = 2, scale_max: int = 100) \
            -> Iterator[Callable[[], "Ising2d"]]:
        """
        Get the 56 default instances of the :class:`Ising2d` problem.

        :param scale_min: the minimum permitted scale, by default `2`
        :param scale_max: the maximum permitted scale, by default `100`
        :returns: a sequence of default :class:`Ising2d` instances

        >>> len(list(Ising2d.default_instances()))
        9

        >>> [x() for x in Ising2d.default_instances()]
        [ising2d_4, ising2d_9, ising2d_16, ising2d_25, ising2d_36, \
ising2d_49, ising2d_64, ising2d_81, ising2d_100]
        """
        return cast(Iterator[Callable[[], "Ising2d"]],
                    super().default_instances(  # type: ignore
                        scale_min, scale_max))
