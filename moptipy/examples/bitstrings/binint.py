"""
The BinInt problem maximizes the binary value of a bit string.

The BinInt problem is similar to the OneMax and LinearHarmonic in that it
tries to maximize the number of `True` bits in a string.
Different from these problems, however, it assigns exponentially increasing
weights to the bits.
The bit at index `1` has weight `2 ** (n - 1)`.
The bit at the last position has weight 1.
The upper bound for the objective function, reached when all bits are `False`,
therefore is `2 ** n - 1`. The lower bound, reached when all bits are `True`,
is `0`.

1. William Michael Rudnick. Genetic Algorithms and Fitness Variance with an
   Application to the Automated Design of Artiﬁcial Neural Networks.
   PhD thesis, Oregon Graduate Institute of Science & Technology: Beaverton,
   OR, USA, 1992. UMI Order No.: GAX92-22642.
2. Dirk Thierens, David Edward Goldberg, and Ângela Guimarães Pereira. Domino
   Convergence, Drift, and the Temporal-Salience Structure of Problems. In
   CEC'98, pages 535-540, 1998.
   doi: https://doi.org/10.1109/ICEC.1998.700085.
3. Kumara Sastry and David Edward Goldberg. Let's Get Ready to Rumble Redux:
   Crossover versus Mutation Head to Head on Exponentially Scaled Problems.
   In GECCO'07-I, pages 1380-1387, 2007.
   doi: https://doi.org10.1145/1276958.1277215.
4. Kumara Sastry and David Edward Goldberg. Let's Get Ready to Rumble Redux:
   Crossover versus Mutation Head to Head on Exponentially Scaled Problems.
   IlliGAL Report 2007005, Illinois Genetic Algorithms Laboratory (IlliGAL),
   Department of Computer Science, Department of General Engineering,
   University of Illinois at Urbana-Champaign: Urbana-Champaign, IL, USA,
   February 11, 2007.
"""

from typing import Callable, Final, Iterator, cast

import numba  # type: ignore
import numpy as np

from moptipy.examples.bitstrings.bitstring_problem import BitStringProblem


@numba.njit(nogil=True, cache=True)
def binint(x: np.ndarray) -> int:
    """
    Get the binint objective value: decode the inverted bit string as integer.

    :param x: the np array
    :return: the inverted bit string represented as integer

    >>> binint(np.array([0]))
    1
    >>> binint(np.array([1]))
    0

    >>> binint(np.array([0, 0]))
    3
    >>> binint(np.array([0, 1]))
    2
    >>> binint(np.array([1, 0]))
    1
    >>> binint(np.array([1, 1]))
    0

    >>> binint(np.array([0, 0, 0]))
    7
    >>> binint(np.array([0, 0, 1]))
    6
    >>> binint(np.array([0, 1, 0]))
    5
    >>> binint(np.array([0, 1, 1]))
    4
    >>> binint(np.array([1, 0, 0]))
    3
    >>> binint(np.array([1, 0, 1]))
    2
    >>> binint(np.array([1, 1, 0]))
    1
    >>> binint(np.array([1, 1, 1]))
    0
    """
    n: Final[int] = len(x)
    weight: int = 1 << n
    result: int = weight - 1
    for xx in x:
        weight >>= 1
        if xx:
            result -= weight
    return result


class BinInt(BitStringProblem):
    """Maximize the binary value of a bit string."""

    def __init__(self, n: int) -> None:
        """
        Initialize the binint objective function.

        :param n: the dimension of the problem

        >>> print(BinInt(2).n)
        2
        >>> print(BinInt(4).evaluate(np.array([True, True, False, True])))
        2
        """
        super().__init__(n)
        self.evaluate = binint  # type: ignore

    def __str__(self) -> str:
        """
        Get the name of the binint objective function.

        :return: `binint_` + length of string

        >>> BinInt(13)
        binint_13
        """
        return f"binint_{self.n}"

    def upper_bound(self) -> int:
        """
        Get the upper bound of the :class:`BinInt` problem.

        :returns: `(1 << n) - 1`

        >>> BinInt(4).upper_bound()
        15
        >>> BinInt(4).evaluate(np.array([0, 0, 0, 0]))
        15
        """
        return (1 << self.n) - 1

    @classmethod
    def default_instances(
            cls: type, scale_min: int = 2, scale_max: int = 30) \
            -> Iterator[Callable[[], "BinInt"]]:
        """
        Get the 29 default instances of the :class:`BinInt` problem.

        :param scale_min: the minimum permitted scale, by default `2`
        :param scale_max: the maximum permitted scale, by default `32`
        :returns: a sequence of default :class:`BinInt` instances

        >>> len(list(BinInt.default_instances()))
        29

        >>> [x() for x in BinInt.default_instances()]
        [binint_2, binint_3, binint_4, binint_5, binint_6, binint_7, \
binint_8, binint_9, binint_10, binint_11, binint_12, binint_13, binint_14, \
binint_15, binint_16, binint_17, binint_18, binint_19, binint_20, binint_21, \
binint_22, binint_23, binint_24, binint_25, binint_26, binint_27, binint_28, \
binint_29, binint_30]
        """
        return cast("Iterator[Callable[[], BinInt]]",
                    super().default_instances(  # type: ignore
                        scale_min, scale_max))
