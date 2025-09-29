"""
An objective function counting the leading ones in a bit string.

LeadingOnes is a standard benchmark problem in evolutionary computation.
It attempts to maximize the number of consecutive True bits at the beginning
of a bit string. Its best possible objective value is 0, the worst possible
is `n`.

1. L. Darrell Whitley. The GENITOR Algorithm and Selection Pressure: Why
   Rank-Based Allocation of Reproductive Trials is Best. In J. David Schaffer,
   ed., Proceedings of the 3rd International Conference on Genetic Algorithms
   (ICGA'89), June 4-7, 1989, Fairfax, VA, USA, pages 116-121. San Francisco,
   CA, USA: Morgan Kaufmann Publishers Inc. ISBN: 1-55860-066-3
   https://www.researchgate.net/publication/2527551
2. Günter Rudolph, Convergence Properties of Evolutionary Algorithms. Hamburg,
   Germany: Verlag Dr. Kovač, 1997.
3. Peyman Afshani, Manindra Agrawal, Benjamin Doerr, Kasper Green Larsen,
   Kurt Mehlhorn, and Carola Winzen. The Query Complexity of Finding a Hidden
   Permutation. Space-Efficient Data Structures, Streams, and Algorithms.
   pp. 1-11. Springer. 2013.
   doi: https://doi.org/10.1007/978-3-642-40273-9_1
4. Peyman Afshani, Manindra Agrawal, Benjamin Doerr, Carola Doerr, Kasper
   Green Larsen, Kurt Mehlhorn. The Query Complexity of a Permutation-Based
   Variant of Mastermind. Discrete Applied Mathematics. 260:28-50. 2019.
   doi: https://doi.org/10.1016/j.dam.2019.01.007
5. Stefan Droste, Thomas Jansen, and Ingo Wegener. On the Analysis of the
   (1+1) Evolutionary Algorithm. Theoretical Computer Science.
   276(1-2):51-81. 2002.
   doi: https://doi.org/10.1016/S0304-3975(01)00182-7.
6. Denis Antipov, Benjamin Doerr, and Vitalii Karavaev. A Tight Runtime
   Analysis for the (1+(λ,λ)) GA on LeadingOnes. FOGA 2019, pp. 169-182. ACM.
   doi: https://doi.org/10.1145/3299904.3340317.
7. Vitalii Karavaev, Denis Antipov, and Benjamin Doerr. Theoretical and
   Empirical Study of the (1+(λ,λ)) EA on the LeadingOnes Problem. GECCO
   (Companion) 2019. pp 2036-2039. ACM.
   doi: https://doi.org/10.1145/3319619.3326910.
8. Thomas Weise, Zhize Wu, Xinlu Li, and Yan Chen. Frequency Fitness
   Assignment: Making Optimization Algorithms Invariant under Bijective
   Transformations of the Objective Function Value. *IEEE Transactions on
   Evolutionary Computation* 25(2):307-319. April 2021. Preprint available at
   arXiv:2001.01416v5 [cs.NE] 15 Oct 2020.
   https://dx.doi.org/10.1109/TEVC.2020.3032090
9. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
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
def leadingones(x: np.ndarray) -> int:
    """
    Get the length of the string minus the number of leading ones.

    :param x: the np array
    :return: the number of leading ones

    >>> leadingones(np.array([False, False, True, False, False]))
    5
    >>> leadingones(np.array([True, False, False, True, True]))
    4
    >>> leadingones(np.array([True, True, False, False, False]))
    3
    >>> leadingones(np.array([True, True, True, False, True]))
    2
    >>> leadingones(np.array([True, True, True, True, False]))
    1
    >>> leadingones(np.array([True, True, True, True, True]))
    0

    # n = 1 and 0 true bits
    >>> leadingones(np.array([0]))
    1

    # n = 1 and 1 true bit
    >>> leadingones(np.array([1]))
    0

    # n = 2 and 0 true bits
    >>> leadingones(np.array([0, 0]))
    2

    # n = 2 and 1 true bit
    >>> leadingones(np.array([1, 0]))
    1

    # n = 2 and 1 true bit
    >>> leadingones(np.array([0, 1]))
    2

    # n = 2 and 1 true bit
    >>> leadingones(np.array([1, 0]))
    1

    # n = 2 and 2 true bits
    >>> leadingones(np.array([1, 1]))
    0

    # n = 3 and 0 true bits
    >>> leadingones(np.array([0, 0, 0]))
    3

    # n = 3 and 1 true bit
    >>> leadingones(np.array([0, 1, 0]))
    3

    # n = 3 and 1 true bit
    >>> leadingones(np.array([0, 0, 1]))
    3

    # n = 3 and 1 true bit
    >>> leadingones(np.array([0, 1, 0]))
    3

    # n = 3 and 2 true bits
    >>> leadingones(np.array([1, 1, 0]))
    1

    # n = 3 and 2 true bits
    >>> leadingones(np.array([1, 0, 1]))
    2

    # n = 3 and 2 true bits
    >>> leadingones(np.array([1, 1, 0]))
    1

    # n = 3 and 3 true bits
    >>> leadingones(np.array([1, 1, 1]))
    0

    # n = 4 and 0 true bits
    >>> leadingones(np.array([0, 0, 0, 0]))
    4

    # n = 4 and 1 true bit
    >>> leadingones(np.array([1, 0, 0, 0]))
    3

    # n = 4 and 1 true bit
    >>> leadingones(np.array([0, 0, 1, 0]))
    4

    # n = 4 and 1 true bit
    >>> leadingones(np.array([1, 0, 0, 0]))
    3

    # n = 4 and 2 true bits
    >>> leadingones(np.array([1, 0, 0, 1]))
    3

    # n = 4 and 2 true bits
    >>> leadingones(np.array([0, 1, 0, 1]))
    4

    # n = 4 and 2 true bits
    >>> leadingones(np.array([0, 1, 1, 0]))
    4

    # n = 4 and 3 true bits
    >>> leadingones(np.array([1, 1, 1, 0]))
    1

    # n = 4 and 3 true bits
    >>> leadingones(np.array([1, 1, 1, 0]))
    1

    # n = 4 and 3 true bits
    >>> leadingones(np.array([0, 1, 1, 1]))
    4

    # n = 4 and 4 true bits
    >>> leadingones(np.array([1, 1, 1, 1]))
    0

    # n = 5 and 0 true bits
    >>> leadingones(np.array([0, 0, 0, 0, 0]))
    5

    # n = 5 and 1 true bit
    >>> leadingones(np.array([0, 1, 0, 0, 0]))
    5

    # n = 5 and 1 true bit
    >>> leadingones(np.array([0, 1, 0, 0, 0]))
    5

    # n = 5 and 1 true bit
    >>> leadingones(np.array([0, 1, 0, 0, 0]))
    5

    # n = 5 and 2 true bits
    >>> leadingones(np.array([1, 1, 0, 0, 0]))
    3

    # n = 5 and 2 true bits
    >>> leadingones(np.array([1, 0, 0, 1, 0]))
    4

    # n = 5 and 2 true bits
    >>> leadingones(np.array([0, 0, 1, 1, 0]))
    5

    # n = 5 and 3 true bits
    >>> leadingones(np.array([1, 0, 0, 1, 1]))
    4

    # n = 5 and 3 true bits
    >>> leadingones(np.array([1, 1, 1, 0, 0]))
    2

    # n = 5 and 3 true bits
    >>> leadingones(np.array([0, 0, 1, 1, 1]))
    5

    # n = 5 and 4 true bits
    >>> leadingones(np.array([1, 1, 0, 1, 1]))
    3

    # n = 5 and 4 true bits
    >>> leadingones(np.array([1, 1, 0, 1, 1]))
    3

    # n = 5 and 4 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 0]))
    1

    # n = 5 and 5 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 1]))
    0

    # n = 6 and 0 true bits
    >>> leadingones(np.array([0, 0, 0, 0, 0, 0]))
    6

    # n = 6 and 1 true bit
    >>> leadingones(np.array([0, 0, 1, 0, 0, 0]))
    6

    # n = 6 and 1 true bit
    >>> leadingones(np.array([0, 1, 0, 0, 0, 0]))
    6

    # n = 6 and 1 true bit
    >>> leadingones(np.array([1, 0, 0, 0, 0, 0]))
    5

    # n = 6 and 2 true bits
    >>> leadingones(np.array([0, 1, 0, 0, 0, 1]))
    6

    # n = 6 and 2 true bits
    >>> leadingones(np.array([0, 0, 1, 0, 0, 1]))
    6

    # n = 6 and 2 true bits
    >>> leadingones(np.array([0, 0, 1, 0, 1, 0]))
    6

    # n = 6 and 3 true bits
    >>> leadingones(np.array([0, 1, 0, 1, 1, 0]))
    6

    # n = 6 and 3 true bits
    >>> leadingones(np.array([1, 0, 0, 0, 1, 1]))
    5

    # n = 6 and 3 true bits
    >>> leadingones(np.array([0, 0, 1, 1, 1, 0]))
    6

    # n = 6 and 4 true bits
    >>> leadingones(np.array([0, 1, 0, 1, 1, 1]))
    6

    # n = 6 and 4 true bits
    >>> leadingones(np.array([0, 1, 1, 1, 1, 0]))
    6

    # n = 6 and 4 true bits
    >>> leadingones(np.array([0, 1, 1, 0, 1, 1]))
    6

    # n = 6 and 5 true bits
    >>> leadingones(np.array([1, 0, 1, 1, 1, 1]))
    5

    # n = 6 and 5 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 1, 0]))
    1

    # n = 6 and 5 true bits
    >>> leadingones(np.array([1, 0, 1, 1, 1, 1]))
    5

    # n = 6 and 6 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 1, 1]))
    0

    # n = 7 and 0 true bits
    >>> leadingones(np.array([0, 0, 0, 0, 0, 0, 0]))
    7

    # n = 7 and 1 true bit
    >>> leadingones(np.array([0, 0, 0, 0, 0, 0, 1]))
    7

    # n = 7 and 1 true bit
    >>> leadingones(np.array([0, 0, 0, 0, 1, 0, 0]))
    7

    # n = 7 and 1 true bit
    >>> leadingones(np.array([0, 0, 1, 0, 0, 0, 0]))
    7

    # n = 7 and 2 true bits
    >>> leadingones(np.array([0, 1, 0, 0, 0, 1, 0]))
    7

    # n = 7 and 2 true bits
    >>> leadingones(np.array([0, 0, 1, 0, 0, 1, 0]))
    7

    # n = 7 and 2 true bits
    >>> leadingones(np.array([0, 0, 0, 1, 0, 1, 0]))
    7

    # n = 7 and 3 true bits
    >>> leadingones(np.array([1, 0, 0, 0, 0, 1, 1]))
    6

    # n = 7 and 3 true bits
    >>> leadingones(np.array([1, 0, 1, 0, 0, 1, 0]))
    6

    # n = 7 and 3 true bits
    >>> leadingones(np.array([1, 0, 0, 1, 1, 0, 0]))
    6

    # n = 7 and 4 true bits
    >>> leadingones(np.array([1, 0, 0, 1, 1, 1, 0]))
    6

    # n = 7 and 4 true bits
    >>> leadingones(np.array([1, 0, 1, 1, 0, 1, 0]))
    6

    # n = 7 and 4 true bits
    >>> leadingones(np.array([1, 1, 1, 0, 1, 0, 0]))
    4

    # n = 7 and 5 true bits
    >>> leadingones(np.array([0, 1, 1, 1, 0, 1, 1]))
    7

    # n = 7 and 5 true bits
    >>> leadingones(np.array([1, 1, 0, 0, 1, 1, 1]))
    5

    # n = 7 and 5 true bits
    >>> leadingones(np.array([0, 1, 1, 1, 1, 0, 1]))
    7

    # n = 7 and 6 true bits
    >>> leadingones(np.array([1, 1, 0, 1, 1, 1, 1]))
    5

    # n = 7 and 6 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 0, 1, 1]))
    3

    # n = 7 and 6 true bits
    >>> leadingones(np.array([1, 1, 0, 1, 1, 1, 1]))
    5

    # n = 7 and 7 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 8 and 0 true bits
    >>> leadingones(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    8

    # n = 8 and 1 true bit
    >>> leadingones(np.array([0, 0, 0, 0, 0, 1, 0, 0]))
    8

    # n = 8 and 1 true bit
    >>> leadingones(np.array([0, 0, 0, 0, 1, 0, 0, 0]))
    8

    # n = 8 and 1 true bit
    >>> leadingones(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    7

    # n = 8 and 2 true bits
    >>> leadingones(np.array([0, 0, 0, 1, 0, 0, 0, 1]))
    8

    # n = 8 and 2 true bits
    >>> leadingones(np.array([0, 1, 0, 0, 0, 0, 0, 1]))
    8

    # n = 8 and 2 true bits
    >>> leadingones(np.array([0, 1, 0, 1, 0, 0, 0, 0]))
    8

    # n = 8 and 3 true bits
    >>> leadingones(np.array([1, 0, 0, 0, 1, 0, 1, 0]))
    7

    # n = 8 and 3 true bits
    >>> leadingones(np.array([1, 0, 1, 0, 1, 0, 0, 0]))
    7

    # n = 8 and 3 true bits
    >>> leadingones(np.array([0, 0, 0, 1, 0, 0, 1, 1]))
    8

    # n = 8 and 4 true bits
    >>> leadingones(np.array([1, 1, 0, 0, 0, 0, 1, 1]))
    6

    # n = 8 and 4 true bits
    >>> leadingones(np.array([1, 1, 0, 1, 0, 1, 0, 0]))
    6

    # n = 8 and 4 true bits
    >>> leadingones(np.array([0, 0, 1, 0, 1, 1, 1, 0]))
    8

    # n = 8 and 5 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 0, 0, 1, 0]))
    4

    # n = 8 and 5 true bits
    >>> leadingones(np.array([1, 0, 1, 1, 0, 0, 1, 1]))
    7

    # n = 8 and 5 true bits
    >>> leadingones(np.array([0, 1, 0, 1, 1, 1, 0, 1]))
    8

    # n = 8 and 6 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 0, 1, 0, 1]))
    4

    # n = 8 and 6 true bits
    >>> leadingones(np.array([1, 1, 0, 1, 1, 1, 0, 1]))
    6

    # n = 8 and 6 true bits
    >>> leadingones(np.array([1, 1, 0, 1, 1, 1, 0, 1]))
    6

    # n = 8 and 7 true bits
    >>> leadingones(np.array([0, 1, 1, 1, 1, 1, 1, 1]))
    8

    # n = 8 and 7 true bits
    >>> leadingones(np.array([1, 1, 0, 1, 1, 1, 1, 1]))
    6

    # n = 8 and 7 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 1, 0, 1, 1]))
    3

    # n = 8 and 8 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 9 and 0 true bits
    >>> leadingones(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
    9

    # n = 9 and 1 true bit
    >>> leadingones(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]))
    9

    # n = 9 and 1 true bit
    >>> leadingones(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]))
    9

    # n = 9 and 1 true bit
    >>> leadingones(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]))
    9

    # n = 9 and 2 true bits
    >>> leadingones(np.array([0, 0, 0, 0, 0, 1, 0, 0, 1]))
    9

    # n = 9 and 2 true bits
    >>> leadingones(np.array([0, 0, 0, 0, 1, 1, 0, 0, 0]))
    9

    # n = 9 and 2 true bits
    >>> leadingones(np.array([1, 0, 0, 0, 0, 0, 1, 0, 0]))
    8

    # n = 9 and 3 true bits
    >>> leadingones(np.array([0, 1, 0, 0, 1, 0, 0, 0, 1]))
    9

    # n = 9 and 3 true bits
    >>> leadingones(np.array([0, 1, 1, 0, 0, 1, 0, 0, 0]))
    9

    # n = 9 and 3 true bits
    >>> leadingones(np.array([0, 1, 1, 0, 0, 0, 0, 0, 1]))
    9

    # n = 9 and 4 true bits
    >>> leadingones(np.array([0, 1, 0, 0, 1, 0, 0, 1, 1]))
    9

    # n = 9 and 4 true bits
    >>> leadingones(np.array([1, 1, 0, 0, 1, 0, 0, 1, 0]))
    7

    # n = 9 and 4 true bits
    >>> leadingones(np.array([0, 1, 1, 1, 0, 0, 1, 0, 0]))
    9

    # n = 9 and 5 true bits
    >>> leadingones(np.array([0, 0, 1, 1, 1, 1, 0, 1, 0]))
    9

    # n = 9 and 5 true bits
    >>> leadingones(np.array([0, 0, 1, 1, 1, 0, 0, 1, 1]))
    9

    # n = 9 and 5 true bits
    >>> leadingones(np.array([1, 0, 0, 1, 1, 0, 0, 1, 1]))
    8

    # n = 9 and 6 true bits
    >>> leadingones(np.array([1, 1, 1, 0, 1, 1, 0, 0, 1]))
    6

    # n = 9 and 6 true bits
    >>> leadingones(np.array([1, 0, 1, 1, 1, 1, 1, 0, 0]))
    8

    # n = 9 and 6 true bits
    >>> leadingones(np.array([1, 1, 0, 1, 1, 1, 0, 0, 1]))
    7

    # n = 9 and 7 true bits
    >>> leadingones(np.array([1, 1, 0, 1, 0, 1, 1, 1, 1]))
    7

    # n = 9 and 7 true bits
    >>> leadingones(np.array([1, 0, 1, 1, 1, 1, 0, 1, 1]))
    8

    # n = 9 and 7 true bits
    >>> leadingones(np.array([1, 0, 1, 1, 1, 0, 1, 1, 1]))
    8

    # n = 9 and 8 true bits
    >>> leadingones(np.array([1, 1, 1, 0, 1, 1, 1, 1, 1]))
    6

    # n = 9 and 8 true bits
    >>> leadingones(np.array([1, 0, 1, 1, 1, 1, 1, 1, 1]))
    8

    # n = 9 and 8 true bits
    >>> leadingones(np.array([1, 0, 1, 1, 1, 1, 1, 1, 1]))
    8

    # n = 9 and 9 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
    0

    # n = 10 and 0 true bits
    >>> leadingones(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    10

    # n = 10 and 1 true bit
    >>> leadingones(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
    10

    # n = 10 and 1 true bit
    >>> leadingones(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    9

    # n = 10 and 1 true bit
    >>> leadingones(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    10

    # n = 10 and 2 true bits
    >>> leadingones(np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0]))
    10

    # n = 10 and 2 true bits
    >>> leadingones(np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0]))
    10

    # n = 10 and 2 true bits
    >>> leadingones(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]))
    10

    # n = 10 and 3 true bits
    >>> leadingones(np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 0]))
    10

    # n = 10 and 3 true bits
    >>> leadingones(np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0]))
    9

    # n = 10 and 3 true bits
    >>> leadingones(np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1]))
    10

    # n = 10 and 4 true bits
    >>> leadingones(np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 0]))
    8

    # n = 10 and 4 true bits
    >>> leadingones(np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0]))
    9

    # n = 10 and 4 true bits
    >>> leadingones(np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1]))
    9

    # n = 10 and 5 true bits
    >>> leadingones(np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1]))
    10

    # n = 10 and 5 true bits
    >>> leadingones(np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0]))
    10

    # n = 10 and 5 true bits
    >>> leadingones(np.array([0, 1, 1, 1, 1, 0, 0, 1, 0, 0]))
    10

    # n = 10 and 6 true bits
    >>> leadingones(np.array([1, 0, 1, 1, 1, 1, 1, 0, 0, 0]))
    9

    # n = 10 and 6 true bits
    >>> leadingones(np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0]))
    7

    # n = 10 and 6 true bits
    >>> leadingones(np.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 1]))
    7

    # n = 10 and 7 true bits
    >>> leadingones(np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, 0]))
    10

    # n = 10 and 7 true bits
    >>> leadingones(np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 0]))
    10

    # n = 10 and 7 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 1, 0, 0, 1, 0, 1]))
    5

    # n = 10 and 8 true bits
    >>> leadingones(np.array([1, 0, 1, 1, 1, 1, 1, 1, 0, 1]))
    9

    # n = 10 and 8 true bits
    >>> leadingones(np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1]))
    7

    # n = 10 and 8 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]))
    6

    # n = 10 and 9 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1]))
    6

    # n = 10 and 9 true bits
    >>> leadingones(np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1]))
    9

    # n = 10 and 9 true bits
    >>> leadingones(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    10

    # n = 10 and 10 true bits
    >>> leadingones(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    0
    """
    xlen: Final[int] = len(x)
    for i in range(xlen):
        if not x[i]:
            return xlen - i
    return 0


class LeadingOnes(BitStringProblem):
    """Maximize the number of leading ones in a bit string."""

    def __init__(self, n: int) -> None:
        """
        Initialize the leading ones objective function.

        :param n: the dimension of the problem

        >>> LeadingOnes(55).n
        55
        >>> LeadingOnes(4).evaluate(np.array([True, True, True, False]))
        1
        """
        super().__init__(n)
        self.evaluate = leadingones  # type: ignore

    def __str__(self) -> str:
        """
        Get the name of the leadingones objective function.

        :return: `leadingones_` + lenth of string

        >>> LeadingOnes(10)
        leadingones_10
        """
        return f"leadingones_{self.n}"

    @classmethod
    def default_instances(
            cls: type, scale_min: int = 2, scale_max: int = 4096) \
            -> Iterator[Callable[[], "LeadingOnes"]]:
        """
        Get the 163 default instances of the :class:`LeadingOnes` problem.

        :param scale_min: the minimum permitted scale, by default `2`
        :param scale_max: the maximum permitted scale, by default `4096`
        :returns: a sequence of default :class:`LeadingOnes` instances

        >>> len(list(LeadingOnes.default_instances()))
        163

        >>> [x() for x in LeadingOnes.default_instances()]
        [leadingones_2, leadingones_3, leadingones_4, leadingones_5, \
leadingones_6, leadingones_7, leadingones_8, leadingones_9, leadingones_10, \
leadingones_11, leadingones_12, leadingones_13, leadingones_14, \
leadingones_15, leadingones_16, leadingones_17, leadingones_18, \
leadingones_19, leadingones_20, leadingones_21, leadingones_22, \
leadingones_23, leadingones_24, leadingones_25, leadingones_26, \
leadingones_27, leadingones_28, leadingones_29, leadingones_30, \
leadingones_31, leadingones_32, leadingones_33, leadingones_36, \
leadingones_40, leadingones_41, leadingones_42, leadingones_44, \
leadingones_48, leadingones_49, leadingones_50, leadingones_55, \
leadingones_59, leadingones_60, leadingones_64, leadingones_66, \
leadingones_70, leadingones_77, leadingones_79, leadingones_80, \
leadingones_81, leadingones_85, leadingones_88, leadingones_90, \
leadingones_96, leadingones_99, leadingones_100, leadingones_107, \
leadingones_111, leadingones_121, leadingones_125, leadingones_128, \
leadingones_144, leadingones_149, leadingones_169, leadingones_170, \
leadingones_192, leadingones_196, leadingones_199, leadingones_200, \
leadingones_222, leadingones_225, leadingones_243, leadingones_256, \
leadingones_269, leadingones_289, leadingones_300, leadingones_324, \
leadingones_333, leadingones_341, leadingones_343, leadingones_359, \
leadingones_361, leadingones_384, leadingones_400, leadingones_441, \
leadingones_444, leadingones_479, leadingones_484, leadingones_500, \
leadingones_512, leadingones_529, leadingones_555, leadingones_576, \
leadingones_600, leadingones_625, leadingones_641, leadingones_666, \
leadingones_676, leadingones_682, leadingones_700, leadingones_729, \
leadingones_768, leadingones_777, leadingones_784, leadingones_800, \
leadingones_841, leadingones_857, leadingones_888, leadingones_900, \
leadingones_961, leadingones_999, leadingones_1000, leadingones_1024, \
leadingones_1089, leadingones_1111, leadingones_1151, leadingones_1156, \
leadingones_1225, leadingones_1296, leadingones_1365, leadingones_1369, \
leadingones_1444, leadingones_1521, leadingones_1536, leadingones_1543, \
leadingones_1600, leadingones_1681, leadingones_1764, leadingones_1849, \
leadingones_1936, leadingones_2000, leadingones_2025, leadingones_2048, \
leadingones_2063, leadingones_2116, leadingones_2187, leadingones_2209, \
leadingones_2222, leadingones_2304, leadingones_2401, leadingones_2500, \
leadingones_2601, leadingones_2704, leadingones_2730, leadingones_2753, \
leadingones_2809, leadingones_2916, leadingones_3000, leadingones_3025, \
leadingones_3072, leadingones_3125, leadingones_3136, leadingones_3249, \
leadingones_3333, leadingones_3364, leadingones_3481, leadingones_3600, \
leadingones_3671, leadingones_3721, leadingones_3844, leadingones_3969, \
leadingones_4000, leadingones_4096]
        """
        return cast("Iterator[Callable[[], LeadingOnes]]",
                    super().default_instances(  # type: ignore
                        scale_min, scale_max))
