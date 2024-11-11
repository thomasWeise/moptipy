"""
A base class for bitstring-based problems.

Many benchmark problems from discrete optimization are simple functions
defined over bit strings. We here offer the class :class:`BitStringProblem`,
which provides reasonable default behavior and several utilities for
implementing such problems.
"""

from math import isqrt
from typing import Final

from pycommons.types import check_int_range

from moptipy.api.objective import Objective
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.logger import KeyValueLogSection


class BitStringProblem(Objective):
    """
    A base class for problems defined over bit strings.

    This base class has a set of default behaviors. It has an attribute
    :attr:`n` denoting the lengths of the accepted bit strings. Its
    :meth:`lower_bound` returns `0` and its :meth:`upper_bound` returns
    :attr:`n`. :meth:`is_always_integer` returns `True`. If also offers
    the method :meth:`space` which returns an instance of
    :class:`~moptipy.spaces.bitstrings.BitStrings` for bit strings of
    length :attr:`n`.

    >>> bs = BitStringProblem(1)
    >>> bs.n
    1

    >>> try:
    ...     bs = BitStringProblem(0)
    ... except ValueError as ve:
    ...     print(ve)
    n=0 is invalid, must be in 1..1000000000.

    >>> try:
    ...     bs = BitStringProblem("a")
    ... except TypeError as te:
    ...     print(te)
    n should be an instance of int but is str, namely 'a'.
    """

    def __init__(self, n: int) -> None:  # +book
        """
        Initialize the bitstring objective function.

        :param n: the dimension of the problem
        """
        super().__init__()
        #: the length of the bit strings
        self.n: Final[int] = check_int_range(n, "n", 1)

    def lower_bound(self) -> int:
        """
        Get the lower bound of the bit string based problem.

        By default, this method returns `0`. Problems where the lower bound
        differs should override this method.

        :return: 0

        >>> print(BitStringProblem(10).lower_bound())
        0
        """
        return 0

    def upper_bound(self) -> int:
        """
        Get the upper bound of the bit string based problem.

        The default value is the length of the bit string. Problems where the
        upper bound differs should overrrite this method.

        :return: by default, this is the length of the bit string

        >>> print(BitStringProblem(7).upper_bound())
        7
        """
        return self.n

    def is_always_integer(self) -> bool:
        """
        Return `True` if the `evaluate` function always returns an `int`.

        This pre-defined function for bit-string based problems will always
        return `True`. Problems where this is not the case should overwrite
        this method.

        :retval True: always
        """
        return True

    def space(self) -> BitStrings:
        """
        Create a proper search space for this problem.

        :returns: an instance of
            :class:`~moptipy.spaces.bitstrings.BitStrings` for bit strings of
            length :attr:`n`
        """
        return BitStrings(self.n)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         BitStringProblem(5).log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[1]
        'name: bitstringproblem_5'
        >>> text[3]
        'lowerBound: 0'
        >>> text[4]
        'upperBound: 5'
        >>> text[5]
        'n: 5'
        >>> len(text)
        7
        """
        super().log_parameters_to(logger)
        logger.key_value("n", self.n)

    def __str__(self) -> str:
        """
        Get the name of the problem.

        :returns: the name of the problem, which by default is the class name
            in lower case, followed by an underscore and the number of bits
        """
        return f"{super().__str__().lower()}_{self.n}"


class SquareBitStringProblem(BitStringProblem):
    """
    A bitstring problem which requires that the string length is square.

    >>> sb = SquareBitStringProblem(9)
    >>> sb.n
    9
    >>> sb.k
    3

    >>> try:
    ...     bs = SquareBitStringProblem(0)
    ... except ValueError as ve:
    ...     print(ve)
    n=0 is invalid, must be in 4..1000000000.

    >>> try:
    ...     bs = SquareBitStringProblem(3)
    ... except ValueError as ve:
    ...     print(ve)
    n=3 is invalid, must be in 4..1000000000.

    >>> try:
    ...     bs = SquareBitStringProblem(21)
    ... except ValueError as ve:
    ...     print(ve)
    n=21 must be a square number, but isqrt(n)=4 does not satisfy n = k*k.

    >>> try:
    ...     bs = SquareBitStringProblem("a")
    ... except TypeError as te:
    ...     print(te)
    n should be an instance of int but is str, namely 'a'.
    """

    def __init__(self, n: int) -> None:
        """
        Initialize the square bitstring problem.

        :param n: the dimension of the problem (must be a perfect square)
        """
        super().__init__(check_int_range(n, "n", 4))
        k: Final[int] = check_int_range(isqrt(n), "k", 2)
        if (k * k) != n:
            raise ValueError(f"n={n} must be a square number, but"
                             f" isqrt(n)={k} does not satisfy n = k*k.")
        #: the k value, i.e., the number of bits per row and column of
        #: the square
        self.k: Final[int] = k

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         SquareBitStringProblem(49).log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[1]
        'name: squarebitstringproblem_49'
        >>> text[3]
        'lowerBound: 0'
        >>> text[4]
        'upperBound: 49'
        >>> text[5]
        'n: 49'
        >>> text[6]
        'k: 7'
        >>> len(text)
        8
        """
        super().log_parameters_to(logger)
        logger.key_value("k", self.k)


class BitStringNKProblem(BitStringProblem):
    """
    A bit string problem with a second parameter `k` with `1 < k < n/2`.

    >>> sb = BitStringNKProblem(9, 3)
    >>> sb.n
    9
    >>> sb.k
    3

    >>> try:
    ...     bs = BitStringNKProblem(0, 3)
    ... except ValueError as ve:
    ...     print(ve)
    n=0 is invalid, must be in 6..1000000000.

    >>> try:
    ...     bs = BitStringNKProblem(5, 2)
    ... except ValueError as ve:
    ...     print(ve)
    n=5 is invalid, must be in 6..1000000000.

    >>> try:
    ...     bs = BitStringNKProblem(21, 20)
    ... except ValueError as ve:
    ...     print(ve)
    k=20 is invalid, must be in 2..9.

    >>> try:
    ...     bs = BitStringNKProblem("a", 3)
    ... except TypeError as te:
    ...     print(te)
    n should be an instance of int but is str, namely 'a'.

    >>> try:
    ...     bs = BitStringNKProblem(13, "x")
    ... except TypeError as te:
    ...     print(te)
    k should be an instance of int but is str, namely 'x'.
    """

    def __init__(self, n: int, k: int) -> None:  # +book
        """
        Initialize the n-k objective function.

        :param n: the dimension of the problem
        :param k: the second parameter
        """
        super().__init__(check_int_range(n, "n", 6))
        #: the second parameter, with `1 < k < n/2`
        self.k: Final[int] = check_int_range(k, "k", 2, (n >> 1) - 1)

    def __str__(self) -> str:
        """
        Get the name of the objective function.

        :return: `class_` + length of string + `_` + k

        >>> BitStringNKProblem(13, 4)
        bitstringnkproblem_13_4
        """
        return f"{super().__str__()}_{self.k}"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         BitStringNKProblem(23, 7).log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[1]
        'name: bitstringnkproblem_23_7'
        >>> text[3]
        'lowerBound: 0'
        >>> text[4]
        'upperBound: 23'
        >>> text[5]
        'n: 23'
        >>> text[6]
        'k: 7'
        >>> len(text)
        8
        """
        super().log_parameters_to(logger)
        logger.key_value("k", self.k)
