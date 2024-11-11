"""
The TwoMax problem.

TwoMax has two optima. The local optimum at the string with all `False` bits.
The objective value of this local optimum is 1. The global optimum is the
string with all `True` bits and its objective value is 0. The worst objective
value is reached if half of the bits are `True` and the other half is `False`.
Then, the objective value will be `(n // 2) + 1`.

The TwoMax problem is based on OneMax but introduces deceptiveness in the
objective function by having a local and a global optimum. Since their basins
of attraction have the same size, a (1 + 1) EA can solve the problem in
`Omega(n ln n)` steps with probability 0.5 while otherwise needing exponential
runtime in expectation, leading to a total expected runtime in
`Omega(n ** n)`.

1. Tobias Friedrich, Francesco Quinzan, and Markus Wagner. Escaping Large
   Deceptive Basins of Attraction with Heavy-Tailed Mutation Operators.
   GECCO 2018. ACM.
   doi: https://doi.org/10.1145/3205455.3205515.
2. Clarissa Van Hoyweghen, David E. Goldberg, and Bart Naudts. From TwoMax to
   the Ising Model: Easy and Hard Symmetrical Problems. GECCO 2002.
   pp 626-633. Morgan Kaufmann.
3. Tobias Friedrich, Pietro S. Oliveto, Dirk Sudholt, and Carsten Witt.
   Analysis of Diversity-Preserving Mechanisms for Global Exploration.
   Evolutionary Computation. 17(4):455-476. 2009.
   doi: https://doi.org/10.1162/evco.2009.17.4.17401.
4. Thomas Weise, Zhize Wu, Xinlu Li, and Yan Chen. Frequency Fitness
   Assignment: Making Optimization Algorithms Invariant under Bijective
   Transformations of the Objective Function Value. *IEEE Transactions on
   Evolutionary Computation* 25(2):307-319. April 2021. Preprint available at
   arXiv:2001.01416v5 [cs.NE] 15 Oct 2020.
   https://dx.doi.org/10.1109/TEVC.2020.3032090
5. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
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

from moptipy.examples.bitstrings.bitstring_problem import BitStringProblem


@numba.njit(nogil=True, cache=True)
def twomax(x: np.ndarray) -> int:
    """
    Compute the objective value of the TwoMax problem.

    :param x: the np array
    :return: the sum of the indices of the first two ones

    >>> twomax(np.array([True, True, True, True, True]))
    0
    >>> twomax(np.array([False, False, False, False, False]))
    1
    >>> twomax(np.array([False, True, True, False, False]))
    3
    >>> twomax(np.array([True, True, True, True, True]))
    0
    >>> twomax(np.array([False, False, False, False, False]))
    1
    >>> twomax(np.array([False, True, False, False, False]))
    2
    >>> twomax(np.array([False, False, False, False, False, False]))
    1
    >>> twomax(np.array([False, False, True, False, False, False]))
    2
    >>> twomax(np.array([False, True, False, True, False, False]))
    3
    >>> twomax(np.array([False, True, False, True, False, True]))
    4
    >>> twomax(np.array([True, False, True, False, True, True]))
    3
    >>> twomax(np.array([False, True, True, True, True, True]))
    2
    >>> twomax(np.array([True, True, True, True, True, True]))
    0

    # n = 1 and 0 true bits
    >>> twomax(np.array([0]))
    1

    # n = 1 and 1 true bit
    >>> twomax(np.array([1]))
    0

    # n = 2 and 0 true bits
    >>> twomax(np.array([0, 0]))
    1

    # n = 2 and 1 true bit
    >>> twomax(np.array([0, 1]))
    2

    # n = 2 and 1 true bit
    >>> twomax(np.array([0, 1]))
    2

    # n = 2 and 1 true bit
    >>> twomax(np.array([0, 1]))
    2

    # n = 2 and 2 true bits
    >>> twomax(np.array([1, 1]))
    0

    # n = 3 and 0 true bits
    >>> twomax(np.array([0, 0, 0]))
    1

    # n = 3 and 1 true bit
    >>> twomax(np.array([1, 0, 0]))
    2

    # n = 3 and 1 true bit
    >>> twomax(np.array([0, 1, 0]))
    2

    # n = 3 and 1 true bit
    >>> twomax(np.array([0, 0, 1]))
    2

    # n = 3 and 2 true bits
    >>> twomax(np.array([1, 0, 1]))
    2

    # n = 3 and 2 true bits
    >>> twomax(np.array([0, 1, 1]))
    2

    # n = 3 and 2 true bits
    >>> twomax(np.array([1, 1, 0]))
    2

    # n = 3 and 3 true bits
    >>> twomax(np.array([1, 1, 1]))
    0

    # n = 4 and 0 true bits
    >>> twomax(np.array([0, 0, 0, 0]))
    1

    # n = 4 and 1 true bit
    >>> twomax(np.array([1, 0, 0, 0]))
    2

    # n = 4 and 1 true bit
    >>> twomax(np.array([0, 0, 0, 1]))
    2

    # n = 4 and 1 true bit
    >>> twomax(np.array([1, 0, 0, 0]))
    2

    # n = 4 and 2 true bits
    >>> twomax(np.array([0, 1, 1, 0]))
    3

    # n = 4 and 2 true bits
    >>> twomax(np.array([0, 1, 0, 1]))
    3

    # n = 4 and 2 true bits
    >>> twomax(np.array([0, 0, 1, 1]))
    3

    # n = 4 and 3 true bits
    >>> twomax(np.array([1, 0, 1, 1]))
    2

    # n = 4 and 3 true bits
    >>> twomax(np.array([1, 1, 1, 0]))
    2

    # n = 4 and 3 true bits
    >>> twomax(np.array([1, 1, 0, 1]))
    2

    # n = 4 and 4 true bits
    >>> twomax(np.array([1, 1, 1, 1]))
    0

    # n = 5 and 0 true bits
    >>> twomax(np.array([0, 0, 0, 0, 0]))
    1

    # n = 5 and 1 true bit
    >>> twomax(np.array([0, 0, 0, 1, 0]))
    2

    # n = 5 and 1 true bit
    >>> twomax(np.array([0, 1, 0, 0, 0]))
    2

    # n = 5 and 1 true bit
    >>> twomax(np.array([1, 0, 0, 0, 0]))
    2

    # n = 5 and 2 true bits
    >>> twomax(np.array([0, 1, 1, 0, 0]))
    3

    # n = 5 and 2 true bits
    >>> twomax(np.array([0, 1, 0, 0, 1]))
    3

    # n = 5 and 2 true bits
    >>> twomax(np.array([1, 1, 0, 0, 0]))
    3

    # n = 5 and 3 true bits
    >>> twomax(np.array([1, 1, 0, 1, 0]))
    3

    # n = 5 and 3 true bits
    >>> twomax(np.array([0, 1, 0, 1, 1]))
    3

    # n = 5 and 3 true bits
    >>> twomax(np.array([1, 0, 1, 1, 0]))
    3

    # n = 5 and 4 true bits
    >>> twomax(np.array([0, 1, 1, 1, 1]))
    2

    # n = 5 and 4 true bits
    >>> twomax(np.array([0, 1, 1, 1, 1]))
    2

    # n = 5 and 4 true bits
    >>> twomax(np.array([1, 1, 0, 1, 1]))
    2

    # n = 5 and 5 true bits
    >>> twomax(np.array([1, 1, 1, 1, 1]))
    0

    # n = 6 and 0 true bits
    >>> twomax(np.array([0, 0, 0, 0, 0, 0]))
    1

    # n = 6 and 1 true bit
    >>> twomax(np.array([0, 0, 0, 0, 0, 1]))
    2

    # n = 6 and 1 true bit
    >>> twomax(np.array([1, 0, 0, 0, 0, 0]))
    2

    # n = 6 and 1 true bit
    >>> twomax(np.array([0, 0, 1, 0, 0, 0]))
    2

    # n = 6 and 2 true bits
    >>> twomax(np.array([1, 1, 0, 0, 0, 0]))
    3

    # n = 6 and 2 true bits
    >>> twomax(np.array([0, 0, 0, 1, 0, 1]))
    3

    # n = 6 and 2 true bits
    >>> twomax(np.array([1, 0, 0, 0, 1, 0]))
    3

    # n = 6 and 3 true bits
    >>> twomax(np.array([0, 1, 1, 0, 1, 0]))
    4

    # n = 6 and 3 true bits
    >>> twomax(np.array([0, 1, 1, 0, 0, 1]))
    4

    # n = 6 and 3 true bits
    >>> twomax(np.array([1, 0, 1, 0, 1, 0]))
    4

    # n = 6 and 4 true bits
    >>> twomax(np.array([1, 0, 0, 1, 1, 1]))
    3

    # n = 6 and 4 true bits
    >>> twomax(np.array([1, 0, 1, 1, 0, 1]))
    3

    # n = 6 and 4 true bits
    >>> twomax(np.array([1, 0, 0, 1, 1, 1]))
    3

    # n = 6 and 5 true bits
    >>> twomax(np.array([0, 1, 1, 1, 1, 1]))
    2

    # n = 6 and 5 true bits
    >>> twomax(np.array([1, 1, 0, 1, 1, 1]))
    2

    # n = 6 and 5 true bits
    >>> twomax(np.array([1, 1, 0, 1, 1, 1]))
    2

    # n = 6 and 6 true bits
    >>> twomax(np.array([1, 1, 1, 1, 1, 1]))
    0

    # n = 7 and 0 true bits
    >>> twomax(np.array([0, 0, 0, 0, 0, 0, 0]))
    1

    # n = 7 and 1 true bit
    >>> twomax(np.array([0, 0, 0, 0, 0, 1, 0]))
    2

    # n = 7 and 1 true bit
    >>> twomax(np.array([0, 1, 0, 0, 0, 0, 0]))
    2

    # n = 7 and 1 true bit
    >>> twomax(np.array([0, 0, 0, 0, 0, 1, 0]))
    2

    # n = 7 and 2 true bits
    >>> twomax(np.array([0, 0, 1, 0, 0, 1, 0]))
    3

    # n = 7 and 2 true bits
    >>> twomax(np.array([1, 1, 0, 0, 0, 0, 0]))
    3

    # n = 7 and 2 true bits
    >>> twomax(np.array([0, 1, 0, 0, 0, 0, 1]))
    3

    # n = 7 and 3 true bits
    >>> twomax(np.array([0, 1, 1, 0, 0, 0, 1]))
    4

    # n = 7 and 3 true bits
    >>> twomax(np.array([1, 0, 0, 0, 1, 1, 0]))
    4

    # n = 7 and 3 true bits
    >>> twomax(np.array([1, 0, 0, 0, 1, 1, 0]))
    4

    # n = 7 and 4 true bits
    >>> twomax(np.array([1, 1, 0, 0, 1, 1, 0]))
    4

    # n = 7 and 4 true bits
    >>> twomax(np.array([1, 1, 0, 1, 0, 0, 1]))
    4

    # n = 7 and 4 true bits
    >>> twomax(np.array([1, 1, 0, 1, 1, 0, 0]))
    4

    # n = 7 and 5 true bits
    >>> twomax(np.array([1, 1, 1, 0, 1, 0, 1]))
    3

    # n = 7 and 5 true bits
    >>> twomax(np.array([1, 1, 1, 0, 0, 1, 1]))
    3

    # n = 7 and 5 true bits
    >>> twomax(np.array([1, 1, 0, 1, 0, 1, 1]))
    3

    # n = 7 and 6 true bits
    >>> twomax(np.array([1, 1, 0, 1, 1, 1, 1]))
    2

    # n = 7 and 6 true bits
    >>> twomax(np.array([1, 1, 1, 1, 0, 1, 1]))
    2

    # n = 7 and 6 true bits
    >>> twomax(np.array([1, 1, 1, 0, 1, 1, 1]))
    2

    # n = 7 and 7 true bits
    >>> twomax(np.array([1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 8 and 0 true bits
    >>> twomax(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    1

    # n = 8 and 1 true bit
    >>> twomax(np.array([0, 0, 0, 0, 0, 0, 0, 1]))
    2

    # n = 8 and 1 true bit
    >>> twomax(np.array([0, 0, 0, 0, 0, 0, 1, 0]))
    2

    # n = 8 and 1 true bit
    >>> twomax(np.array([0, 0, 0, 0, 1, 0, 0, 0]))
    2

    # n = 8 and 2 true bits
    >>> twomax(np.array([0, 1, 0, 1, 0, 0, 0, 0]))
    3

    # n = 8 and 2 true bits
    >>> twomax(np.array([0, 1, 0, 0, 1, 0, 0, 0]))
    3

    # n = 8 and 2 true bits
    >>> twomax(np.array([0, 0, 0, 0, 1, 0, 0, 1]))
    3

    # n = 8 and 3 true bits
    >>> twomax(np.array([0, 0, 0, 0, 1, 0, 1, 1]))
    4

    # n = 8 and 3 true bits
    >>> twomax(np.array([0, 0, 0, 0, 0, 1, 1, 1]))
    4

    # n = 8 and 3 true bits
    >>> twomax(np.array([0, 0, 0, 1, 1, 1, 0, 0]))
    4

    # n = 8 and 4 true bits
    >>> twomax(np.array([1, 1, 0, 0, 1, 1, 0, 0]))
    5

    # n = 8 and 4 true bits
    >>> twomax(np.array([1, 1, 0, 0, 0, 1, 1, 0]))
    5

    # n = 8 and 4 true bits
    >>> twomax(np.array([0, 1, 1, 0, 1, 0, 1, 0]))
    5

    # n = 8 and 5 true bits
    >>> twomax(np.array([1, 1, 0, 1, 0, 1, 1, 0]))
    4

    # n = 8 and 5 true bits
    >>> twomax(np.array([0, 1, 0, 1, 0, 1, 1, 1]))
    4

    # n = 8 and 5 true bits
    >>> twomax(np.array([1, 1, 0, 0, 1, 1, 1, 0]))
    4

    # n = 8 and 6 true bits
    >>> twomax(np.array([1, 1, 1, 1, 0, 1, 1, 0]))
    3

    # n = 8 and 6 true bits
    >>> twomax(np.array([1, 1, 1, 1, 0, 1, 0, 1]))
    3

    # n = 8 and 6 true bits
    >>> twomax(np.array([0, 1, 1, 1, 1, 1, 1, 0]))
    3

    # n = 8 and 7 true bits
    >>> twomax(np.array([1, 1, 1, 1, 0, 1, 1, 1]))
    2

    # n = 8 and 7 true bits
    >>> twomax(np.array([1, 1, 1, 0, 1, 1, 1, 1]))
    2

    # n = 8 and 7 true bits
    >>> twomax(np.array([1, 1, 1, 1, 1, 1, 0, 1]))
    2

    # n = 8 and 8 true bits
    >>> twomax(np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 9 and 0 true bits
    >>> twomax(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
    1

    # n = 9 and 1 true bit
    >>> twomax(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]))
    2

    # n = 9 and 1 true bit
    >>> twomax(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]))
    2

    # n = 9 and 1 true bit
    >>> twomax(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]))
    2

    # n = 9 and 2 true bits
    >>> twomax(np.array([0, 0, 0, 1, 0, 0, 0, 1, 0]))
    3

    # n = 9 and 2 true bits
    >>> twomax(np.array([0, 0, 0, 0, 0, 1, 0, 0, 1]))
    3

    # n = 9 and 2 true bits
    >>> twomax(np.array([0, 1, 0, 0, 0, 0, 0, 1, 0]))
    3

    # n = 9 and 3 true bits
    >>> twomax(np.array([0, 1, 0, 0, 0, 0, 1, 0, 1]))
    4

    # n = 9 and 3 true bits
    >>> twomax(np.array([0, 1, 0, 1, 0, 0, 0, 0, 1]))
    4

    # n = 9 and 3 true bits
    >>> twomax(np.array([0, 0, 1, 0, 0, 1, 0, 1, 0]))
    4

    # n = 9 and 4 true bits
    >>> twomax(np.array([1, 1, 0, 0, 0, 0, 0, 1, 1]))
    5

    # n = 9 and 4 true bits
    >>> twomax(np.array([0, 0, 1, 0, 0, 1, 1, 1, 0]))
    5

    # n = 9 and 4 true bits
    >>> twomax(np.array([0, 1, 0, 1, 0, 1, 0, 0, 1]))
    5

    # n = 9 and 5 true bits
    >>> twomax(np.array([1, 1, 1, 0, 0, 0, 0, 1, 1]))
    5

    # n = 9 and 5 true bits
    >>> twomax(np.array([0, 0, 0, 1, 1, 1, 0, 1, 1]))
    5

    # n = 9 and 5 true bits
    >>> twomax(np.array([0, 0, 1, 1, 0, 1, 1, 1, 0]))
    5

    # n = 9 and 6 true bits
    >>> twomax(np.array([1, 1, 0, 1, 1, 0, 0, 1, 1]))
    4

    # n = 9 and 6 true bits
    >>> twomax(np.array([1, 0, 1, 1, 0, 1, 1, 0, 1]))
    4

    # n = 9 and 6 true bits
    >>> twomax(np.array([1, 0, 1, 1, 1, 0, 1, 0, 1]))
    4

    # n = 9 and 7 true bits
    >>> twomax(np.array([1, 1, 1, 1, 1, 0, 0, 1, 1]))
    3

    # n = 9 and 7 true bits
    >>> twomax(np.array([1, 1, 1, 0, 0, 1, 1, 1, 1]))
    3

    # n = 9 and 7 true bits
    >>> twomax(np.array([1, 0, 1, 1, 1, 0, 1, 1, 1]))
    3

    # n = 9 and 8 true bits
    >>> twomax(np.array([1, 1, 0, 1, 1, 1, 1, 1, 1]))
    2

    # n = 9 and 8 true bits
    >>> twomax(np.array([1, 1, 1, 1, 1, 0, 1, 1, 1]))
    2

    # n = 9 and 8 true bits
    >>> twomax(np.array([1, 1, 1, 1, 1, 1, 0, 1, 1]))
    2

    # n = 9 and 9 true bits
    >>> twomax(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 10 and 0 true bits
    >>> twomax(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    1

    # n = 10 and 1 true bit
    >>> twomax(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
    2

    # n = 10 and 1 true bit
    >>> twomax(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    2

    # n = 10 and 1 true bit
    >>> twomax(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    2

    # n = 10 and 2 true bits
    >>> twomax(np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
    3

    # n = 10 and 2 true bits
    >>> twomax(np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0]))
    3

    # n = 10 and 2 true bits
    >>> twomax(np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0]))
    3

    # n = 10 and 3 true bits
    >>> twomax(np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1]))
    4

    # n = 10 and 3 true bits
    >>> twomax(np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 1]))
    4

    # n = 10 and 3 true bits
    >>> twomax(np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0]))
    4

    # n = 10 and 4 true bits
    >>> twomax(np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0]))
    5

    # n = 10 and 4 true bits
    >>> twomax(np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1]))
    5

    # n = 10 and 4 true bits
    >>> twomax(np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0]))
    5

    # n = 10 and 5 true bits
    >>> twomax(np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 0]))
    6

    # n = 10 and 5 true bits
    >>> twomax(np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 0]))
    6

    # n = 10 and 5 true bits
    >>> twomax(np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 0]))
    6

    # n = 10 and 6 true bits
    >>> twomax(np.array([1, 1, 1, 1, 0, 0, 1, 0, 0, 1]))
    5

    # n = 10 and 6 true bits
    >>> twomax(np.array([0, 1, 0, 1, 1, 1, 1, 0, 0, 1]))
    5

    # n = 10 and 6 true bits
    >>> twomax(np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 0]))
    5

    # n = 10 and 7 true bits
    >>> twomax(np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1]))
    4

    # n = 10 and 7 true bits
    >>> twomax(np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 0]))
    4

    # n = 10 and 7 true bits
    >>> twomax(np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 1]))
    4

    # n = 10 and 8 true bits
    >>> twomax(np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1]))
    3

    # n = 10 and 8 true bits
    >>> twomax(np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0]))
    3

    # n = 10 and 8 true bits
    >>> twomax(np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]))
    3

    # n = 10 and 9 true bits
    >>> twomax(np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1]))
    2

    # n = 10 and 9 true bits
    >>> twomax(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    2

    # n = 10 and 9 true bits
    >>> twomax(np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1]))
    2

    # n = 10 and 10 true bits
    >>> twomax(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    0
    """
    s: Final[int] = len(x)
    number_of_ones: Final[int] = int(x.sum())
    return 0 if s == number_of_ones else (
        1 + s - max(number_of_ones, s - number_of_ones))


class TwoMax(BitStringProblem):
    """The TwoMax benchmark problem."""

    def __init__(self, n: int) -> None:  # +book
        """
        Initialize the twomax objective function.

        :param n: the dimension of the problem

        >>> TwoMax(2).n
        2
        >>> TwoMax(4).evaluate(np.array([True, True, False, True]))
        2
        """
        super().__init__(n)
        self.evaluate = twomax  # type: ignore

    def __str__(self) -> str:
        """
        Get the name of the twomax objective function.

        :return: `twomax_` + length of string

        >>> print(TwoMax(13))
        twomax_13
        """
        return f"twomax_{self.n}"

    def upper_bound(self) -> int:
        """
        Get the upper bound of the twomax problem.

        :return: the length of the bit string integer-divided by 2, plus 1

        >>> TwoMax(15).upper_bound()
        8

        >>> TwoMax(5).upper_bound()
        3
        """
        return (self.n // 2) + 1
