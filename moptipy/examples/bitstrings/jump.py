"""
The Jump problem.

The jump problem is basically OneMax, but with a deceptive region of k bit
flips before the optimum.

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
"""

from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.examples.bitstrings.bitstring_problem import BitStringProblem
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import check_int_range


@numba.njit(nogil=True, cache=True)
def jump(x: np.ndarray, k: int) -> int:
    """
    Compute the jump value.

    :param x: the np array
    :param k: the k parameter
    :return: jump value

    >>> print(jump(np.array([False, False, False, False, False, False]), 2))
    6
    >>> print(jump(np.array([False, False, False, False, True, False]), 2))
    5
    >>> print(jump(np.array([False, True, True, False, False, False]), 2))
    4
    >>> print(jump(np.array([True, False, True, False, True, False]), 2))
    3
    >>> print(jump(np.array([True, False, True, False, True, True]), 2))
    2
    >>> print(jump(np.array([True, True, True, True, True, False]), 2))
    7
    >>> print(jump(np.array([True, True, True, True, True, True]), 2))
    0
    """
    res: Final[int] = x.sum()
    n: Final[int] = len(x)
    nmk: Final[int] = n - k
    if (res >= n) or (res <= nmk):
        return int(n - res)
    return int(k + res)


class Jump(BitStringProblem):
    """Compute the Jump problem."""

    def __init__(self, n: int, k: int) -> None:  # +book
        """
        Initialize the jump objective function.

        :param n: the dimension of the problem
        :param k: the jump length
        """
        super().__init__(n)
        #: the jump width
        self.k: Final[int] = check_int_range(k, "k", 2, (n >> 1) - 1)

    def __str__(self) -> str:
        """
        Get the name of the jump objective function.

        :return: `jump_` + length of string + `_` + k

        >>> print(Jump(13, 4))
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

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         Jump(23, 7).log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[1]
        'name: jump_23_7'
        >>> text[3]
        'lowerBound: 0'
        >>> text[4]
        'upperBound: 29'
        >>> text[5]
        'n: 23'
        >>> text[6]
        'k: 7'
        >>> len(text)
        8
        """
        super().log_parameters_to(logger)
        logger.key_value("k", self.k)

    def upper_bound(self) -> int:
        """
        Get the upper bound of the jump problem.

        :return: the length of the bit string

        >>> print(Jump(15, 4).upper_bound())
        18
        """
        return self.n + self.k - 1
