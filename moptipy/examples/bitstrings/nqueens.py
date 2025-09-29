"""
The N-Queens problem.

The N-Queens problem is defined for bit strings of length n = N ** 2.
A bit string `x` is mapped to a chess board and a queen is placed for any bit
of value 1. The goal is to place N queens such that they cannot attack each
other. The total number of queens on the board be Q(x) is the number of `True`
bits, which might be more or less than N.
We also count the number ez(x) of queens in every single row, column, and
diagonal z of the chess board. The minimization version of the N-Queens
problem is then `N - Q(x) + N sum_(all z) max(0, ez(x) - 1)`.
The N-Queens problems are attested moderate difficulty in [1].

Here, `N` is stored in the `k`-value of the benchmark function instances.

The best possible objective value of this function is achieved if exactly
`k=N` queens are positioned such that none can beat any other queen. The
objective function then returns 0.

The worst case is if a queen is placed on every single field, i.e., if we have
`n = k*k` queens on the field. Then, the objective value will be
`(((((k - 2) * 4) + 1) * k) + 3) * k`.

1. Carola Doerr, Furong Ye, Naama Horesh, Hao Wang, Ofer M. Shir, and Thomas
   Bäck. Benchmarking Discrete Optimization Heuristics with IOHprofiler.
   Applied Soft Computing Journal. 88:106027. 2020.
   doi: https://doi.org/10.1016/j.asoc.2019.106027
2. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
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
"""
from typing import Callable, Final, Iterator, cast

import numba  # type: ignore
import numpy as np
from pycommons.types import check_int_range

from moptipy.examples.bitstrings.bitstring_problem import (
    SquareBitStringProblem,
)


@numba.njit(nogil=True, cache=True)
def nqueens(x: np.ndarray, k: int) -> int:
    """
    Evaluate the N-Queens objective function.

    :param x: the np array representing the board (bit string)
    :param k: the total number of queens (dimension of the problem)
    :return: the penalty score

    >>> nqueens(np.array([False,  True, False, False,
    ...                   False, False, False,  True,
    ...                    True, False, False, False,
    ...                   False, False,  True, False]), 4)
    0

    >>> nqueens(np.array([False, False,  True, False,
    ...                    True, False, False, False,
    ...                   False, False, False,  True,
    ...                   False,  True, False, False]), 4)
    0

    >>> nqueens(np.array([False, False, False, False,
    ...                   False, False, False, False,
    ...                   False, False, False, False,
    ...                   False, False, False, False]), 4)
    4

    >>> nqueens(np.array([ True, False, False, False,
    ...                   False, False, False, False,
    ...                   False, False, False, False,
    ...                   False, False, False, False]), 4)
    3

    # two queens, but in the same row, which gives 2 + 4
    >>> nqueens(np.array([ True,  True, False, False,
    ...                   False, False, False, False,
    ...                   False, False, False, False,
    ...                   False, False, False, False]), 4)
    6

    # three queens, but 2 in the same row, 2 in the same column, and 2 in one
    # diagonal, which gives 1 + 4 + 4 + 4
    >>> nqueens(np.array([ True,  True, False, False,
    ...                    True, False, False, False,
    ...                   False, False, False, False,
    ...                   False, False, False, False]), 4)
    13

    # four queens, but 3 in the same row, 2 in the same column, and 2 in one
    # diagonal, which gives 0 + 8 + 4 + 4
    >>> nqueens(np.array([ True,  True,  True, False,
    ...                    True, False, False, False,
    ...                   False, False, False, False,
    ...                   False, False, False, False]), 4)
    16

    # five queens, but 4 in the same row, 2 in the same column, and 2 in one
    # diagonal, which gives -1 + 12 + 4 + 4
    >>> nqueens(np.array([ True,  True,  True,  True,
    ...                    True, False, False, False,
    ...                   False, False, False, False,
    ...                   False, False, False, False]), 4)
    19

    >>> nqueens(np.array([
    ...     False, False, False, True, False, False, False, False,
    ...     False, False, False, False, False, True, False, False,
    ...     False, False, False, False, False, False, False, True,
    ...     False, True, False, False, False, False, False, False,
    ...     False, False, False, False, False, False, True, False,
    ...     True, False, False, False, False, False, False, False,
    ...     False, False, True, False, False, False, False, False,
    ...     False, False, False, False, True, False, False, False,]), 8)
    0

    # five queens, but 4 in the same row, 2 in the same column, and 2 in one
    # diagonal, which gives -1 + 12 + 4 + 4
    >>> nqueens(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4)
    19

    # 16 bits, board width = 4, and 2 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 4)
    6

    # 16 bits, board width = 4, and 2 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 4)
    2

    # 16 bits, board width = 4, and 2 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]), 4)
    2

    # 16 bits, board width = 4, and 3 queens
    >>> nqueens(np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 4)
    5

    # 16 bits, board width = 4, and 4 queens
    >>> nqueens(np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 4)
    12

    # 16 bits, board width = 4, and 4 queens
    >>> nqueens(np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 4)
    12

    # 16 bits, board width = 4, and 4 queens
    >>> nqueens(np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]), 4)
    12

    # 16 bits, board width = 4, and 4 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]), 4)
    20

    # 16 bits, board width = 4, and 11 queens
    >>> nqueens(np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1]), 4)
    85

    # 16 bits, board width = 4, and 8 queens
    >>> nqueens(np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]), 4)
    52

    # 16 bits, board width = 4, and 9 queens
    >>> nqueens(np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]), 4)
    67

    # 16 bits, board width = 4, and 10 queens
    >>> nqueens(np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]), 4)
    78

    # 16 bits, board width = 4, and 0 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4)
    4

    # 16 bits, board width = 4, and 16 queens
    >>> nqueens(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 4)
    156

    # 25 bits, board width = 5, and 3 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 0, 1, 0, 0, 0, 0, 0]), 5)
    17

    # 25 bits, board width = 5, and 1 queen
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0]), 5)
    4

    # 25 bits, board width = 5, and 4 queens
    >>> nqueens(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ...    0, 0, 0, 0, 1, 1, 0, 0]), 5)
    16

    # 25 bits, board width = 5, and 1 queen
    >>> nqueens(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0]), 5)
    4

    # 25 bits, board width = 5, and 5 queens
    >>> nqueens(np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 1, 0, 0, 1, 0]), 5)
    25

    # 25 bits, board width = 5, and 5 queens
    >>> nqueens(np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0]), 5)
    30

    # 25 bits, board width = 5, and 5 queens
    >>> nqueens(np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 1]), 5)
    15

    # 25 bits, board width = 5, and 5 queens
    >>> nqueens(np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
    ...    0, 1, 0, 0, 0, 0, 0, 0]), 5)
    20

    # 25 bits, board width = 5, and 15 queens
    >>> nqueens(np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1,
    ...    1, 1, 0, 1, 1, 0, 1, 1]), 5)
    160

    # 25 bits, board width = 5, and 22 queens
    >>> nqueens(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
    ...    1, 1, 1, 1, 1, 1, 1, 1]), 5)
    283

    # 25 bits, board width = 5, and 18 queens
    >>> nqueens(np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 0, 1]), 5)
    217

    # 25 bits, board width = 5, and 21 queens
    >>> nqueens(np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1]), 5)
    274

    # 25 bits, board width = 5, and 0 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0]), 5)
    5

    # 25 bits, board width = 5, and 25 queens
    >>> nqueens(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1]), 5)
    340

    # 36 bits, board width = 6, and 2 queens
    >>> nqueens(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 6)
    4

    # 36 bits, board width = 6, and 2 queens
    >>> nqueens(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 6)
    4

    # 36 bits, board width = 6, and 4 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
    ...    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 6)
    14

    # 36 bits, board width = 6, and 3 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 6)
    3

    # 36 bits, board width = 6, and 6 queens
    >>> nqueens(np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    ...    0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), 6)
    42

    # 36 bits, board width = 6, and 6 queens
    >>> nqueens(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
    ...    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]), 6)
    42

    # 36 bits, board width = 6, and 6 queens
    >>> nqueens(np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]), 6)
    36

    # 36 bits, board width = 6, and 6 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    ...    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]), 6)
    42

    # 36 bits, board width = 6, and 17 queens
    >>> nqueens(np.array([0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0,
    ...    0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]), 6)
    223

    # 36 bits, board width = 6, and 13 queens
    >>> nqueens(np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]), 6)
    137

    # 36 bits, board width = 6, and 20 queens
    >>> nqueens(np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1,
    ...    1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1]), 6)
    274

    # 36 bits, board width = 6, and 11 queens
    >>> nqueens(np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
    ...    1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]), 6)
    103

    # 36 bits, board width = 6, and 0 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 6)
    6

    # 36 bits, board width = 6, and 36 queens
    >>> nqueens(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 6)
    630

    # 49 bits, board width = 7, and 1 queen
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 7)
    6

    # 49 bits, board width = 7, and 2 queens
    >>> nqueens(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 7)
    5

    # 49 bits, board width = 7, and 3 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 7)
    4

    # 49 bits, board width = 7, and 4 queens
    >>> nqueens(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 7)
    17

    # 49 bits, board width = 7, and 7 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    ...    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 7)
    70

    # 49 bits, board width = 7, and 7 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 7)
    35

    # 49 bits, board width = 7, and 7 queens
    >>> nqueens(np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), 7)
    42

    # 49 bits, board width = 7, and 7 queens
    >>> nqueens(np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 7)
    63

    # 49 bits, board width = 7, and 25 queens
    >>> nqueens(np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,
    ...    0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0,
    ...    1, 0, 0, 0, 0, 1, 0, 0, 0, 0]), 7)
    437

    # 49 bits, board width = 7, and 33 queens
    >>> nqueens(np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0,
    ...    0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 0, 1, 0, 1, 1, 0]), 7)
    625

    # 49 bits, board width = 7, and 24 queens
    >>> nqueens(np.array([1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1,
    ...    0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0,
    ...    0, 0, 1, 0, 0, 1, 1, 0, 0, 0]), 7)
    403

    # 49 bits, board width = 7, and 15 queens
    >>> nqueens(np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
    ...    0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), 7)
    209

    # 49 bits, board width = 7, and 0 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 7)
    7

    # 49 bits, board width = 7, and 49 queens
    >>> nqueens(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 7)
    1050

    # 64 bits, board width = 8, and 1 queen
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0]), 8)
    7

    # 64 bits, board width = 8, and 1 queen
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0]), 8)
    7

    # 64 bits, board width = 8, and 3 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0]), 8)
    21

    # 64 bits, board width = 8, and 6 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    ...    0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0]), 8)
    42

    # 64 bits, board width = 8, and 8 queens
    >>> nqueens(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 1, 0]), 8)
    64

    # 64 bits, board width = 8, and 8 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
    ...    0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    ...    0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    ...    0, 0, 0]), 8)
    72

    # 64 bits, board width = 8, and 8 queens
    >>> nqueens(np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ...    0, 1, 0]), 8)
    56

    # 64 bits, board width = 8, and 8 queens
    >>> nqueens(np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0]), 8)
    64

    # 64 bits, board width = 8, and 20 queens
    >>> nqueens(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
    ...    1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1,
    ...    0, 0, 0]), 8)
    348

    # 64 bits, board width = 8, and 46 queens
    >>> nqueens(np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
    ...    1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1,
    ...    0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
    ...    1, 0, 1]), 8)
    1074

    # 64 bits, board width = 8, and 29 queens
    >>> nqueens(np.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
    ...    1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
    ...    1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0,
    ...    0, 0, 1]), 8)
    579

    # 64 bits, board width = 8, and 20 queens
    >>> nqueens(np.array([1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
    ...    0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ...    1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ...    0, 0, 0]), 8)
    300

    # 64 bits, board width = 8, and 0 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0]), 8)
    8

    # 64 bits, board width = 8, and 64 queens
    >>> nqueens(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1]), 8)
    1624

    # 81 bits, board width = 9, and 1 queen
    >>> nqueens(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 9)
    8

    # 81 bits, board width = 9, and 7 queens
    >>> nqueens(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
    ...    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), 9)
    56

    # 81 bits, board width = 9, and 8 queens
    >>> nqueens(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 9)
    73

    # 81 bits, board width = 9, and 3 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]), 9)
    15

    # 81 bits, board width = 9, and 9 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 9)
    81

    # 81 bits, board width = 9, and 9 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 9)
    63

    # 81 bits, board width = 9, and 9 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1,
    ...    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 9)
    90

    # 81 bits, board width = 9, and 9 queens
    >>> nqueens(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 9)
    72

    # 81 bits, board width = 9, and 22 queens
    >>> nqueens(np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
    ...    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1,
    ...    0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
    ...    0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 9)
    428

    # 81 bits, board width = 9, and 43 queens
    >>> nqueens(np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    ...    0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ...    1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1,
    ...    0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0]), 9)
    1091

    # 81 bits, board width = 9, and 18 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
    ...    0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
    ...    0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
    ...    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 9)
    288

    # 81 bits, board width = 9, and 24 queens
    >>> nqueens(np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0,
    ...    1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0,
    ...    1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
    ...    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]), 9)
    444

    # 81 bits, board width = 9, and 0 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 9)
    9

    # 81 bits, board width = 9, and 81 queens
    >>> nqueens(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 9)
    2376

    # 100 bits, board width = 10, and 6 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 10)
    34

    # 100 bits, board width = 10, and 7 queens
    >>> nqueens(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ...    0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]), 10)
    43

    # 100 bits, board width = 10, and 8 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    ...    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 10)
    72

    # 100 bits, board width = 10, and 6 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 10)
    34

    # 100 bits, board width = 10, and 10 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    ...    1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 10)
    140

    # 100 bits, board width = 10, and 10 queens
    >>> nqueens(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    ...    1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 10)
    80

    # 100 bits, board width = 10, and 10 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]), 10)
    130

    # 100 bits, board width = 10, and 10 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
    ...    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
    ...    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    ...    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 10)
    100

    # 100 bits, board width = 10, and 50 queens
    >>> nqueens(np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
    ...    0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0,
    ...    0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,
    ...    1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
    ...    0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0]), 10)
    1430

    # 100 bits, board width = 10, and 42 queens
    >>> nqueens(np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0,
    ...    0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,
    ...    1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
    ...    0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0,
    ...    1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0]), 10)
    1138

    # 100 bits, board width = 10, and 93 queens
    >>> nqueens(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
    ...    1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 10)
    3067

    # 100 bits, board width = 10, and 92 queens
    >>> nqueens(np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
    ...    1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]), 10)
    3018

    # 100 bits, board width = 10, and 0 queens
    >>> nqueens(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 10)
    10

    # 100 bits, board width = 10, and 100 queens
    >>> nqueens(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ...    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 10)
    3330

    >>> nqueens(np.array([True] * 16), 4)
    156
    """
    n: Final[int] = len(x)

    # The chess board is stored row-by-row.
    # Count the total number of queens on the chess board. It should be k.
    queens_total: int = 0

    # Since we need to go through the board row by row anyway, we
    # can also check how many queens are in each row while we are at
    # it.
    must_be_one_1: int = 0  # queens per row
    penalty: int = 0

    # The last row goes from index n-1 to index n-k, the row before
    # from n-k-1 to n-2k, ..., the first row goes from k-1 to 0.
    next_row_reset = k - 1
    for i in range(n):
        if x[i]:
            queens_total += 1
            must_be_one_1 += 1

        if i >= next_row_reset:  # We reached the end of a row.
            next_row_reset += k

            # If there is at most one queen in the row, the penalty is zero.
            # Otherwise, the penalty for the row is number of queens in it
            # minus 1.
            if must_be_one_1 > 1:
                penalty += must_be_one_1 - 1
            must_be_one_1 = 0

    # Count the number of queens in the columns and check if there
    # is more than one queen in a column.
    # col be the column index, it goes from 1 to k.
    for col in range(k):
        must_be_one_1 = 0
        # The cells in column j have indices k-j, 2k-j, 3k-j, ...,
        # k*k-j=n-j in x and we iterate them from the back.
        for i in range(col, n, k):
            if x[i]:
                must_be_one_1 += 1
        # If there is at most one queen in the column, the penalty is zero.
        # Otherwise, the penalty for the column is number of queens in
        # it minus 1.
        if must_be_one_1 > 1:
            penalty += must_be_one_1 - 1

    # There are 1 + 2*(k-2) = 2*k-3 diagonals of any kind.
    # The diagonal in the "middle" is unique, the others can be
    # mirrored.

    # We have two types of diagonals.
    # One goes from top-left to bottom-right, for which we use index
    # i1 and count collisions in must_be_one_1 and must_be_one_2 and whose
    # indices step in k-1 increments.
    # The other one goes from bottom-left to top-right, for which we
    # use index i2 and count collisions in must_be_one_3 and must_be_one_4
    # and whose indices step in k+1 increments.
    # Both have the central, non-mirrored version and the others
    # which are mirrored around the central diagonal.
    diagonal_step_1: Final[int] = k - 1
    diagonal_step_2: Final[int] = k + 1
    other_diagonal_start: Final[int] = n - 1

    # First process unique center diagonal.
    must_be_one_1 = 0
    must_be_one_3: int = 0
    d: int = k - 1
    i1: int = k * d
    i2: int = i1 + diagonal_step_1
    while i1 > 0:
        if x[i1]:
            must_be_one_1 += 1
        if x[i2]:
            must_be_one_3 += 1
        i1 -= diagonal_step_1
        i2 -= diagonal_step_2
    if must_be_one_1 > 1:
        penalty += must_be_one_1 - 1
    if must_be_one_3 > 1:
        penalty += must_be_one_3 - 1

    # Now process the mirrored diagonals
    d -= 1
    while d > 0:
        must_be_one_1 = 0
        must_be_one_3 = 0
        must_be_one_2: int = 0
        must_be_one_4: int = 0

        i1 = k * d
        if i1 > other_diagonal_start:
            i1 -= diagonal_step_1 * (
                (i1 - other_diagonal_start) // diagonal_step_1)
        i2 = i1 + diagonal_step_1

        while i1 > 0:
            if x[i1]:
                must_be_one_1 += 1
            if x[other_diagonal_start - i1]:
                must_be_one_2 += 1
            if x[i2]:
                must_be_one_3 += 1
            if x[other_diagonal_start - i2]:
                must_be_one_4 += 1
            i1 -= diagonal_step_1
            i2 -= diagonal_step_2

        if must_be_one_1 > 1:
            penalty += must_be_one_1 - 1
        if must_be_one_2 > 1:
            penalty += must_be_one_2 - 1
        if must_be_one_3 > 1:
            penalty += must_be_one_3 - 1
        if must_be_one_4 > 1:
            penalty += must_be_one_4 - 1
        d -= 1

    # penalty now holds the total number of collisions in the rows, columns,
    # and all diagonals.
    # queens_total is the number of queens.
    # The minimization version of the IOHprofiler N queens problem is then:
    return k - queens_total + (k * penalty)


class NQueens(SquareBitStringProblem):
    """The N-Queens problem."""

    def __init__(self, n: int) -> None:
        """
        Initialize the n-queens problem.

        :param n: the dimension of the problem (must be a perfect square)
            and at least 16
        """
        super().__init__(check_int_range(n, "n", 16))

    def evaluate(self, x: np.ndarray) -> int:
        """
        Evaluate a solution to the N-Queens problem.

        :param x: the bit string to evaluate
        :returns: the value of the N-Queens problem for the string
        """
        return nqueens(x, self.k)

    def __str__(self) -> str:
        """
        Get the name of the N-Queens problem.

        :return: `NQueens_` + dimension of the board

        >>> NQueens(16)
        nqueens_16
        """
        return f"nqueens_{self.n}"

    def upper_bound(self) -> int:
        """
        Compute the upper bound of the N-Queens objective function.

        >>> NQueens(4 * 4).upper_bound()
        156

        >>> NQueens(5 * 5).upper_bound()
        340

        >>> NQueens(6 * 6).upper_bound()
        630

        >>> NQueens(7 * 7).upper_bound()
        1050

        >>> NQueens(8 * 8).upper_bound()
        1624

        >>> NQueens(9 * 9).upper_bound()
        2376

        >>> NQueens(10 * 10).upper_bound()
        3330

        >>> NQueens(20 * 20).upper_bound()
        29260

        >>> NQueens(30 * 30).upper_bound()
        101790

        >>> NQueens(100 * 100).upper_bound()
        3930300

        >>> NQueens(6241).upper_bound()
        1928706

        >>> NQueens(4225).upper_bound()
        1069120
        """
        k: Final[int] = self.k
        return (((((k - 2) * 4) + 1) * k) + 3) * k

    @classmethod
    def default_instances(
            cls: type, scale_min: int = 16, scale_max: int = 144) \
            -> Iterator[Callable[[], "NQueens"]]:
        """
        Get the 9 default instances of the :class:`NQueens` problem.

        :param scale_min: the minimum permitted scale, by default `16`
        :param scale_max: the maximum permitted scale, by default `144`
        :returns: a sequence of default :class:`NQueens` instances

        >>> len(list(NQueens.default_instances()))
        9

        >>> [x() for x in NQueens.default_instances()]
        [nqueens_16, nqueens_25, nqueens_36, nqueens_49, nqueens_64, \
nqueens_81, nqueens_100, nqueens_121, nqueens_144]
        """
        return cast("Iterator[Callable[[], NQueens]]",
                    super().default_instances(  # type: ignore
                        scale_min, scale_max))
