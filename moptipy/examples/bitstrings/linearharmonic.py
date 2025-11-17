"""
The linear harmonic objective function.

The bit at index `i` for `i` in `0..n-1` has weight `i+1`. This is the penalty
that is incurred if the bit is set to `False`.
The best objective value, 0, is hence obtained if all bits are `True`.
The worst objective value, i.e., `n * (n + 1) // 2`, is obtained if all bits
are `False`.

1. Carola Doerr, Furong Ye, Naama Horesh, Hao Wang, Ofer M. Shir, and Thomas
   Bäck. Benchmarking Discrete Optimization Heuristics with IOHprofiler.
   Applied Soft Computing Journal. 88:106027. 2020.
   doi: https://doi.org/10.1016/j.asoc.2019.106027
2. Stefan Droste, Thomas Jansen, and Ingo Wegener. On the Analysis of the
   (1+1) Evolutionary Algorithm. *Theoretical Computer Science.*
   276(1-2):51-81. April 2002.
   doi: https://doi.org/10.1016/S0304-3975(01)00182-7
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
"""

from typing import Callable, Iterator, cast

import numba  # type: ignore
import numpy as np

from moptipy.examples.bitstrings.bitstring_problem import BitStringProblem


@numba.njit(nogil=True, cache=True)
def linear_harmonic(x: np.ndarray) -> float:
    """
    Evaluate the linear function with harmonic weights.

    :param  x: np array representing the bit string
    :return: the objective value

    >>> linear_harmonic(np.array([True, True, True]))
    0
    >>> linear_harmonic(np.array([False, True, True]))
    1
    >>> linear_harmonic(np.array([True, False, True]))
    2
    >>> linear_harmonic(np.array([True, True, False]))
    3
    >>> linear_harmonic(np.array([False, False, True]))
    3
    >>> linear_harmonic(np.array([False, True, False]))
    4
    >>> linear_harmonic(np.array([True, False, False]))
    5
    >>> linear_harmonic(np.array([False, False, False]))
    6
    >>> (3 * (3 + 1)) // 2
    6

    >>> linear_harmonic(np.array([True, True, True, True]))
    0
    >>> linear_harmonic(np.array([False, True, True, True]))
    1
    >>> linear_harmonic(np.array([True, False, True, True]))
    2
    >>> linear_harmonic(np.array([True, True, False, True]))
    3
    >>> linear_harmonic(np.array([True, True, True, False]))
    4
    >>> linear_harmonic(np.array([False, False, True, True]))
    3
    >>> linear_harmonic(np.array([False, True, False, True]))
    4
    >>> linear_harmonic(np.array([False, True, True, False]))
    5
    >>> linear_harmonic(np.array([True, False, False, True]))
    5
    >>> linear_harmonic(np.array([True, False, True, False]))
    6
    >>> linear_harmonic(np.array([True, True, False, False]))
    7
    >>> linear_harmonic(np.array([False, False, False, True]))
    6
    >>> linear_harmonic(np.array([False, False, True, False]))
    7
    >>> linear_harmonic(np.array([False, True, False, False]))
    8
    >>> linear_harmonic(np.array([True, False, False, False]))
    9
    >>> linear_harmonic(np.array([False, False, False, False]))
    10
    >>> 4 * (4 + 1) // 2
    10

    # n = 1 and 0 true bits
    >>> linear_harmonic(np.array([0]))
    1

    # n = 1 and 1 true bit
    >>> linear_harmonic(np.array([1]))
    0

    # n = 2 and 0 true bits
    >>> linear_harmonic(np.array([0, 0]))
    3

    # n = 2 and 1 true bit
    >>> linear_harmonic(np.array([1, 0]))
    2

    # n = 2 and 1 true bit
    >>> linear_harmonic(np.array([1, 0]))
    2

    # n = 2 and 1 true bit
    >>> linear_harmonic(np.array([1, 0]))
    2

    # n = 2 and 2 true bits
    >>> linear_harmonic(np.array([1, 1]))
    0

    # n = 3 and 0 true bits
    >>> linear_harmonic(np.array([0, 0, 0]))
    6

    # n = 3 and 1 true bit
    >>> linear_harmonic(np.array([0, 1, 0]))
    4

    # n = 3 and 1 true bit
    >>> linear_harmonic(np.array([0, 0, 1]))
    3

    # n = 3 and 1 true bit
    >>> linear_harmonic(np.array([1, 0, 0]))
    5

    # n = 3 and 2 true bits
    >>> linear_harmonic(np.array([0, 1, 1]))
    1

    # n = 3 and 2 true bits
    >>> linear_harmonic(np.array([1, 0, 1]))
    2

    # n = 3 and 2 true bits
    >>> linear_harmonic(np.array([1, 1, 0]))
    3

    # n = 3 and 3 true bits
    >>> linear_harmonic(np.array([1, 1, 1]))
    0

    # n = 4 and 0 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 0]))
    10

    # n = 4 and 1 true bit
    >>> linear_harmonic(np.array([1, 0, 0, 0]))
    9

    # n = 4 and 1 true bit
    >>> linear_harmonic(np.array([1, 0, 0, 0]))
    9

    # n = 4 and 1 true bit
    >>> linear_harmonic(np.array([1, 0, 0, 0]))
    9

    # n = 4 and 2 true bits
    >>> linear_harmonic(np.array([0, 1, 1, 0]))
    5

    # n = 4 and 2 true bits
    >>> linear_harmonic(np.array([0, 1, 0, 1]))
    4

    # n = 4 and 2 true bits
    >>> linear_harmonic(np.array([1, 0, 0, 1]))
    5

    # n = 4 and 3 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 1]))
    3

    # n = 4 and 3 true bits
    >>> linear_harmonic(np.array([0, 1, 1, 1]))
    1

    # n = 4 and 3 true bits
    >>> linear_harmonic(np.array([1, 0, 1, 1]))
    2

    # n = 4 and 4 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1]))
    0

    # n = 5 and 0 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 0, 0]))
    15

    # n = 5 and 1 true bit
    >>> linear_harmonic(np.array([0, 1, 0, 0, 0]))
    13

    # n = 5 and 1 true bit
    >>> linear_harmonic(np.array([1, 0, 0, 0, 0]))
    14

    # n = 5 and 2 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 0, 1]))
    7

    # n = 5 and 2 true bits
    >>> linear_harmonic(np.array([0, 1, 0, 1, 0]))
    9

    # n = 5 and 2 true bits
    >>> linear_harmonic(np.array([1, 0, 0, 1, 0]))
    10

    # n = 5 and 3 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 1, 1]))
    3

    # n = 5 and 3 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 1, 1]))
    3

    # n = 5 and 3 true bits
    >>> linear_harmonic(np.array([1, 0, 1, 0, 1]))
    6

    # n = 5 and 4 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 0]))
    5

    # n = 5 and 4 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 0]))
    5

    # n = 5 and 4 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 0]))
    5

    # n = 5 and 5 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1]))
    0

    # n = 6 and 0 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 0, 0, 0]))
    21

    # n = 6 and 1 true bit
    >>> linear_harmonic(np.array([0, 0, 0, 0, 1, 0]))
    16

    # n = 6 and 1 true bit
    >>> linear_harmonic(np.array([0, 0, 1, 0, 0, 0]))
    18

    # n = 6 and 1 true bit
    >>> linear_harmonic(np.array([0, 0, 0, 0, 0, 1]))
    15

    # n = 6 and 2 true bits
    >>> linear_harmonic(np.array([1, 0, 1, 0, 0, 0]))
    17

    # n = 6 and 2 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 0, 1, 0]))
    13

    # n = 6 and 2 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 0, 1, 1]))
    10

    # n = 6 and 3 true bits
    >>> linear_harmonic(np.array([0, 1, 0, 0, 1, 1]))
    8

    # n = 6 and 3 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 0, 0, 1]))
    12

    # n = 6 and 3 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 0, 1, 0]))
    13

    # n = 6 and 4 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 0, 0]))
    11

    # n = 6 and 4 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 1, 1, 0]))
    9

    # n = 6 and 4 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 0, 0, 1]))
    9

    # n = 6 and 5 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 1, 1, 1]))
    3

    # n = 6 and 5 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 0]))
    6

    # n = 6 and 5 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 0, 1]))
    5

    # n = 6 and 6 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 1]))
    0

    # n = 7 and 0 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 0, 0, 0, 0]))
    28

    # n = 7 and 1 true bit
    >>> linear_harmonic(np.array([0, 0, 1, 0, 0, 0, 0]))
    25

    # n = 7 and 1 true bit
    >>> linear_harmonic(np.array([0, 0, 0, 1, 0, 0, 0]))
    24

    # n = 7 and 1 true bit
    >>> linear_harmonic(np.array([1, 0, 0, 0, 0, 0, 0]))
    27

    # n = 7 and 2 true bits
    >>> linear_harmonic(np.array([0, 1, 0, 0, 0, 1, 0]))
    20

    # n = 7 and 2 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 0, 1, 0, 1]))
    16

    # n = 7 and 2 true bits
    >>> linear_harmonic(np.array([1, 0, 0, 0, 0, 1, 0]))
    21

    # n = 7 and 3 true bits
    >>> linear_harmonic(np.array([0, 1, 0, 0, 0, 1, 1]))
    13

    # n = 7 and 3 true bits
    >>> linear_harmonic(np.array([1, 0, 0, 1, 0, 0, 1]))
    16

    # n = 7 and 3 true bits
    >>> linear_harmonic(np.array([0, 1, 0, 0, 0, 1, 1]))
    13

    # n = 7 and 4 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 1, 0, 0, 1]))
    14

    # n = 7 and 4 true bits
    >>> linear_harmonic(np.array([1, 0, 1, 0, 1, 0, 1]))
    12

    # n = 7 and 4 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 1, 1, 1, 0]))
    10

    # n = 7 and 5 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 0, 0, 1, 1]))
    9

    # n = 7 and 5 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 0, 1, 0]))
    12

    # n = 7 and 5 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 1, 0, 1, 1]))
    8

    # n = 7 and 6 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 1, 0]))
    7

    # n = 7 and 6 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 1, 0]))
    7

    # n = 7 and 6 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 1, 1, 1, 1]))
    3

    # n = 7 and 7 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 8 and 0 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    36

    # n = 8 and 1 true bit
    >>> linear_harmonic(np.array([0, 1, 0, 0, 0, 0, 0, 0]))
    34

    # n = 8 and 1 true bit
    >>> linear_harmonic(np.array([0, 0, 0, 0, 1, 0, 0, 0]))
    31

    # n = 8 and 1 true bit
    >>> linear_harmonic(np.array([0, 1, 0, 0, 0, 0, 0, 0]))
    34

    # n = 8 and 2 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 1, 0, 0, 0, 0]))
    29

    # n = 8 and 2 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 0, 0, 0, 1, 0]))
    26

    # n = 8 and 2 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 0, 0, 1, 0, 1]))
    22

    # n = 8 and 3 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 0, 0, 0, 0, 1]))
    25

    # n = 8 and 3 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 0, 0, 0, 0, 1]))
    25

    # n = 8 and 3 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 0, 1, 1, 0, 1]))
    17

    # n = 8 and 4 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 0, 0, 0, 0, 1]))
    22

    # n = 8 and 4 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 0, 0, 0, 0, 1]))
    22

    # n = 8 and 4 true bits
    >>> linear_harmonic(np.array([1, 0, 1, 0, 0, 1, 0, 1]))
    18

    # n = 8 and 5 true bits
    >>> linear_harmonic(np.array([0, 1, 1, 1, 0, 1, 0, 1]))
    13

    # n = 8 and 5 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 0, 1, 1, 1, 1]))
    7

    # n = 8 and 5 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 0, 0, 1, 0]))
    19

    # n = 8 and 6 true bits
    >>> linear_harmonic(np.array([0, 1, 1, 1, 1, 0, 1, 1]))
    7

    # n = 8 and 6 true bits
    >>> linear_harmonic(np.array([0, 1, 1, 1, 1, 1, 1, 0]))
    9

    # n = 8 and 6 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 0, 1, 1, 0]))
    13

    # n = 8 and 7 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 1, 0, 1]))
    7

    # n = 8 and 7 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 0, 1, 1, 1, 1]))
    4

    # n = 8 and 7 true bits
    >>> linear_harmonic(np.array([1, 0, 1, 1, 1, 1, 1, 1]))
    2

    # n = 8 and 8 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 9 and 0 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
    45

    # n = 9 and 1 true bit
    >>> linear_harmonic(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]))
    43

    # n = 9 and 1 true bit
    >>> linear_harmonic(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]))
    37

    # n = 9 and 1 true bit
    >>> linear_harmonic(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]))
    39

    # n = 9 and 2 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 1, 0, 1, 0, 0, 0]))
    35

    # n = 9 and 2 true bits
    >>> linear_harmonic(np.array([1, 0, 0, 1, 0, 0, 0, 0, 0]))
    40

    # n = 9 and 2 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]))
    42

    # n = 9 and 3 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 0, 1, 0, 0, 1, 1]))
    23

    # n = 9 and 3 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 1, 1, 0, 0, 0, 0]))
    33

    # n = 9 and 3 true bits
    >>> linear_harmonic(np.array([0, 1, 1, 0, 0, 0, 1, 0, 0]))
    33

    # n = 9 and 4 true bits
    >>> linear_harmonic(np.array([1, 0, 1, 0, 0, 1, 0, 1, 0]))
    27

    # n = 9 and 4 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 1, 0, 1, 0, 0, 1]))
    23

    # n = 9 and 4 true bits
    >>> linear_harmonic(np.array([0, 1, 0, 1, 0, 1, 0, 0, 1]))
    24

    # n = 9 and 5 true bits
    >>> linear_harmonic(np.array([1, 0, 0, 1, 0, 1, 1, 0, 1]))
    18

    # n = 9 and 5 true bits
    >>> linear_harmonic(np.array([1, 0, 0, 1, 1, 0, 0, 1, 1]))
    18

    # n = 9 and 5 true bits
    >>> linear_harmonic(np.array([0, 1, 1, 1, 0, 1, 0, 0, 1]))
    21

    # n = 9 and 6 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 0, 1, 1, 1, 1, 0]))
    16

    # n = 9 and 6 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 1, 0, 1, 1, 0, 1]))
    16

    # n = 9 and 6 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 0, 1, 0, 1, 1, 0]))
    19

    # n = 9 and 7 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 1, 0, 1, 0]))
    16

    # n = 9 and 7 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]))
    17

    # n = 9 and 7 true bits
    >>> linear_harmonic(np.array([0, 1, 0, 1, 1, 1, 1, 1, 1]))
    4

    # n = 9 and 8 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 0, 1, 1, 1]))
    6

    # n = 9 and 8 true bits
    >>> linear_harmonic(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1]))
    1

    # n = 9 and 8 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 1, 0, 1, 1]))
    7

    # n = 9 and 9 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 10 and 0 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    55

    # n = 10 and 1 true bit
    >>> linear_harmonic(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))
    46

    # n = 10 and 1 true bit
    >>> linear_harmonic(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
    50

    # n = 10 and 1 true bit
    >>> linear_harmonic(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    53

    # n = 10 and 2 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0]))
    46

    # n = 10 and 2 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0]))
    47

    # n = 10 and 2 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]))
    36

    # n = 10 and 3 true bits
    >>> linear_harmonic(np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 1]))
    37

    # n = 10 and 3 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0]))
    34

    # n = 10 and 3 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 1]))
    38

    # n = 10 and 4 true bits
    >>> linear_harmonic(np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0]))
    33

    # n = 10 and 4 true bits
    >>> linear_harmonic(np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 0]))
    31

    # n = 10 and 4 true bits
    >>> linear_harmonic(np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1]))
    34

    # n = 10 and 5 true bits
    >>> linear_harmonic(np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 0]))
    26

    # n = 10 and 5 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 0]))
    25

    # n = 10 and 5 true bits
    >>> linear_harmonic(np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1]))
    19

    # n = 10 and 6 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1]))
    24

    # n = 10 and 6 true bits
    >>> linear_harmonic(np.array([0, 1, 1, 1, 1, 0, 1, 0, 0, 1]))
    24

    # n = 10 and 6 true bits
    >>> linear_harmonic(np.array([0, 0, 1, 0, 1, 1, 1, 1, 0, 1]))
    16

    # n = 10 and 7 true bits
    >>> linear_harmonic(np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 0]))
    18

    # n = 10 and 7 true bits
    >>> linear_harmonic(np.array([1, 0, 1, 0, 1, 1, 0, 1, 1, 1]))
    13

    # n = 10 and 7 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 0]))
    26

    # n = 10 and 8 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 0]))
    13

    # n = 10 and 8 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1]))
    11

    # n = 10 and 8 true bits
    >>> linear_harmonic(np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 1]))
    8

    # n = 10 and 9 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1]))
    9

    # n = 10 and 9 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1]))
    3

    # n = 10 and 9 true bits
    >>> linear_harmonic(np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1]))
    3

    # n = 10 and 10 true bits
    >>> linear_harmonic(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    0
    """
    result: int = 0  # The sum of the penalty weights
    weight: int = 1  # The current penalty weight.

    for xx in x:  # Iterate over all the bits in the array.
        if not xx:  # If the bit is False, then
            result += weight  # add the penalty weight to the result.
        weight += 1  # Increment the penalty weight by 1.  # noqa: SIM113

    return result  # Return the result.


class LinearHarmonic(BitStringProblem):
    """The objective function of the linear harmonic benchmark problem."""

    def __init__(self, n: int) -> None:
        """
        Initialize the Linear Harmonic problem instance.

        :param n: length of the bit string

        >>> LinearHarmonic(5).evaluate(np.array([
        ...     False, True, False, True]))
        4
        """
        super().__init__(n)
        self.evaluate = linear_harmonic  # type: ignore

    def upper_bound(self) -> int:
        """
        Return the upper bound of the linear harmonic function.

        :returns: `n * (n + 1) // 2`

        >>> LinearHarmonic(5).upper_bound()
        15
        """
        return (self.n * (self.n + 1)) // 2

    def __str__(self) -> str:
        """
        Get the name of the Linear Harmonic problem instance.

        :return: 'linearharmonic_' + n value

        >>> print(LinearHarmonic(8))
        linharm_8
        """
        return f"linharm_{self.n}"

    @classmethod
    def default_instances(
            cls: type, scale_min: int = 2, scale_max: int = 333) \
            -> Iterator[Callable[[], "LinearHarmonic"]]:
        """
        Get the 78 default instances of the :class:`LinearHarmonic` problem.

        :param scale_min: the minimum permitted scale, by default `2`
        :param scale_max: the maximum permitted scale, by default `333`
        :returns: a sequence of default :class:`LinearHarmonic` instances

        >>> len(list(LinearHarmonic.default_instances()))
        78

        >>> [x() for x in LinearHarmonic.default_instances()]
        [linharm_2, linharm_3, linharm_4, linharm_5, linharm_6, linharm_7, \
linharm_8, linharm_9, linharm_10, linharm_11, linharm_12, linharm_13, \
linharm_14, linharm_15, linharm_16, linharm_17, linharm_18, linharm_19, \
linharm_20, linharm_21, linharm_22, linharm_23, linharm_24, linharm_25, \
linharm_26, linharm_27, linharm_28, linharm_29, linharm_30, linharm_31, \
linharm_32, linharm_33, linharm_36, linharm_40, linharm_41, linharm_42, \
linharm_44, linharm_48, linharm_49, linharm_50, linharm_55, linharm_59, \
linharm_60, linharm_64, linharm_66, linharm_70, linharm_77, linharm_79, \
linharm_80, linharm_81, linharm_85, linharm_88, linharm_90, linharm_96, \
linharm_99, linharm_100, linharm_107, linharm_111, linharm_121, linharm_125, \
linharm_128, linharm_144, linharm_149, linharm_169, linharm_170, \
linharm_192, linharm_196, linharm_199, linharm_200, linharm_222, \
linharm_225, linharm_243, linharm_256, linharm_269, linharm_289, linharm_300, \
linharm_324, linharm_333]
        """
        return cast("Iterator[Callable[[], LinearHarmonic]]",
                    super().default_instances(  # type: ignore
                        scale_min, scale_max))
