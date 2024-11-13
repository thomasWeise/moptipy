"""
A base class for bitstring-based problems.

Many benchmark problems from discrete optimization are simple functions
defined over bit strings. We here offer the class :class:`BitStringProblem`,
which provides reasonable default behavior and several utilities for
implementing such problems.
"""
from itertools import takewhile
from math import isqrt
from typing import Callable, Final, Generator, Iterator, cast

from pycommons.ds.sequences import merge_sorted_and_return_unique
from pycommons.math.primes import primes
from pycommons.types import check_int_range

from moptipy.api.objective import Objective
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.logger import KeyValueLogSection


def __powers_of(base: int) -> Generator[int, None, None]:
    """
    Yield all powers of the given `base`.

    :returns: a generator with the powers of `base` greater than 1

    >>> from itertools import takewhile
    >>> list(takewhile(lambda x: x < 100, __powers_of(2)))
    [2, 4, 8, 16, 32, 64]

    >>> list(takewhile(lambda x: x < 10000, __powers_of(10)))
    [10, 100, 1000]
    """
    pp: int = base
    while True:
        yield pp
        pp *= base


def __powers_of_2_div_3() -> Generator[int, None, None]:
    """
    Yield all powers of 2 // 3 > 2.

    :returns: a generator with the powers of 2//3 greater than 1

    >>> from itertools import takewhile
    >>> list(takewhile(lambda x: x < 100, __powers_of_2_div_3()))
    [2, 5, 10, 21, 42, 85]
    """
    pp: int = 8
    while True:
        yield pp // 3
        pp *= 2


def __powers_of_2_mul_3() -> Generator[int, None, None]:
    """
    Yield all powers of 2 * 3 > 2.

    :returns: a generator with the powers of 2 * 3 greater than 1

    >>> from itertools import takewhile
    >>> list(takewhile(lambda x: x < 100, __powers_of_2_mul_3()))
    [3, 6, 12, 24, 48, 96]
    """
    pp: int = 1
    while True:
        yield pp * 3
        pp *= 2


def __primes_13() -> Generator[int, None, None]:
    """
    Yield a sequence of prime numbers which increase by 1/3.

    :return: the sequence of prime numbers

    >>> from itertools import takewhile
    >>> list(takewhile(lambda x: x < 100, __primes_13()))
    [2, 3, 5, 7, 11, 17, 23, 31, 41, 59, 79]
    """
    last: int = -1
    for ret in primes():
        if ((4 * last) // 3) <= ret:
            yield ret
            last = ret


def __1111() -> Generator[int, None, None]:
    """
    Yield numbers like 1, 2, ..., 11, 22, 33, ..., 99, 111, 222, 333, ...

    :returns: yield numbers which are multiples of 1, 11, 111, 1111, etc.

    >>> from itertools import takewhile
    >>> list(takewhile(lambda x: x < 10000, __1111()))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222,\
 333, 444, 555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, 5555, 6666, 7777,\
 8888, 9999]
    """
    base: int = 1
    while True:
        next_base = 10 * base
        yield from range(base, next_base, base)
        base = next_base + 1


def __1000() -> Generator[int, None, None]:
    """
    Yield numbers like 1, 2, ..., 10, 20, 30, ..., 90, 100, 200, 300, ...

    :returns: yield numbers which are multiples of 1, 10, 100, 1000, etc.

    >>> from itertools import takewhile
    >>> list(takewhile(lambda x: x < 10000, __1000()))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200,\
 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000,\
 7000, 8000, 9000]
    """
    base: int = 1
    while True:
        next_base = 10 * base
        yield from range(base, next_base, base)
        base = next_base


def default_square_scale_sequence(minimum: int = 2,
                                  maximum: int = 3333) -> Iterator[int]:
    """
    Get the default sequence of square numbers.

    :param minimum: the smallest permitted value, by default `2`
    :param maximum: the largest permitted value, by default `3333`

    >>> list(default_square_scale_sequence())
    [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, \
324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024, \
1089, 1156, 1225, 1296, 1369, 1444, 1521, 1600, 1681, 1764, 1849, 1936, \
2025, 2116, 2209, 2304, 2401, 2500, 2601, 2704, 2809, 2916, 3025, 3136, 3249]

    >>> list(default_square_scale_sequence(100, 1000))
    [100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, \
576, 625, 676, 729, 784, 841, 900, 961]

    >>> try:
    ...     default_square_scale_sequence(-1)
    ... except ValueError as ve:
    ...     print(ve)
    minimum=-1 is invalid, must be in 1..1000000000.

    >>> try:
    ...     default_square_scale_sequence("2")
    ... except TypeError as te:
    ...     print(te)
    minimum should be an instance of int but is str, namely '2'.

    >>> try:
    ...     default_square_scale_sequence(10, 10)
    ... except ValueError as ve:
    ...     print(ve)
    maximum=10 is invalid, must be in 11..1000000000.

    >>> try:
    ...     default_square_scale_sequence(2, "2")
    ... except TypeError as te:
    ...     print(te)
    maximum should be an instance of int but is str, namely '2'.
    """
    check_int_range(maximum, "maximum", check_int_range(
        minimum, "minimum", 1, 1_000_000_000) + 1, 1_000_000_000)
    return (j for j in (i ** 2 for i in range(max(
        1, isqrt(minimum)), isqrt(maximum) + 1)) if minimum <= j <= maximum)


def default_scale_sequence(minimum: int = 2,
                           maximum: int = 3333) -> Iterator[int]:
    """
    Get the default scales for investigating discrete optimization.

    :param minimum: the smallest permitted value, by default `2`
    :param maximum: the largest permitted value, by default `3333`

    >>> list(default_scale_sequence())
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, \
22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 40, 41, 42, 44, 48, 49, \
50, 55, 59, 60, 64, 66, 70, 77, 79, 80, 81, 85, 88, 90, 96, 99, 100, 107, \
111, 121, 125, 128, 144, 149, 169, 170, 192, 196, 199, 200, 222, 225, 243, \
256, 269, 289, 300, 324, 333, 341, 343, 359, 361, 384, 400, 441, 444, 479, \
484, 500, 512, 529, 555, 576, 600, 625, 641, 666, 676, 682, 700, 729, 768, \
777, 784, 800, 841, 857, 888, 900, 961, 999, 1000, 1024, 1089, 1111, 1151, \
1156, 1225, 1296, 1365, 1369, 1444, 1521, 1536, 1543, 1600, 1681, 1764, \
1849, 1936, 2000, 2025, 2048, 2063, 2116, 2187, 2209, 2222, 2304, 2401, \
2500, 2601, 2704, 2730, 2753, 2809, 2916, 3000, 3025, 3072, 3125, 3136, \
3249, 3333]

    >>> list(default_scale_sequence(10, 100))
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, \
28, 29, 30, 31, 32, 33, 36, 40, 41, 42, 44, 48, 49, 50, 55, 59, 60, 64, 66, \
70, 77, 79, 80, 81, 85, 88, 90, 96, 99, 100]

    >>> list(default_scale_sequence(maximum=100))
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, \
22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 40, 41, 42, 44, 48, 49, \
50, 55, 59, 60, 64, 66, 70, 77, 79, 80, 81, 85, 88, 90, 96, 99, 100]

    >>> list(default_scale_sequence(9000, 10000))
    [9000, 9025, 9216, 9409, 9604, 9801, 9999, 10000]

    >>> try:
    ...     default_scale_sequence(-1)
    ... except ValueError as ve:
    ...     print(ve)
    minimum=-1 is invalid, must be in 1..1000000000.

    >>> try:
    ...     default_scale_sequence("2")
    ... except TypeError as te:
    ...     print(te)
    minimum should be an instance of int but is str, namely '2'.

    >>> try:
    ...     default_scale_sequence(10, 10)
    ... except ValueError as ve:
    ...     print(ve)
    maximum=10 is invalid, must be in 11..1000000000.

    >>> try:
    ...     default_scale_sequence(2, "2")
    ... except TypeError as te:
    ...     print(te)
    maximum should be an instance of int but is str, namely '2'.
    """
    check_int_range(maximum, "maximum", check_int_range(
        minimum, "minimum", 1, 1_000_000_000) + 1, 1_000_000_000)
    return filter(
        lambda x: minimum <= x <= maximum, takewhile(
            lambda x: x <= maximum, merge_sorted_and_return_unique(
                __powers_of(2), __powers_of(3), __powers_of(5), __powers_of(7),
                __powers_of(10), __powers_of_2_div_3(), __powers_of_2_mul_3(),
                __primes_13(), __1111(), __1000(), range(32), (625, ),
                default_square_scale_sequence(minimum, maximum))))


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

    @classmethod
    def default_instances(
            cls: type, scale_min: int = 2, scale_max: int = 3333) \
            -> Iterator[Callable[[], "BitStringProblem"]]:
        """
        Get the default instances of this problem type.

        :param scale_min: the minimum scale
        :param scale_max: the maximum scale
        :return: an :class:`Iterator` with the instances
        """
        return (cast(Callable[[], "BitStringProblem"],
                     lambda __i=i: cls(__i)) for i in default_scale_sequence(
            scale_min, scale_max))


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

    @classmethod
    def default_instances(
            cls: type, scale_min: int = 4, scale_max: int = 3333) \
            -> Iterator[Callable[[], "SquareBitStringProblem"]]:
        """
        Get the default instances of this problem type.

        :return: an :class:`Iterator` with the instances
        """
        return (cast(Callable[[], "SquareBitStringProblem"],
                     lambda __i=i: cls(__i))
                for i in default_square_scale_sequence(scale_min, scale_max))


def default_nk_k_sequence(n: int) -> Iterator[int]:
    """
    Get the default values of `k` for a :class:`BitStringNKProblem`.

    :param n: the `n` value for the :class:`BitStringNKProblem`.
    :return: a sequence of appropriate `k` values

    >>> list(default_nk_k_sequence(6))
    [2]

    >>> list(default_nk_k_sequence(7))
    [2]

    >>> list(default_nk_k_sequence(8))
    [2, 3]

    >>> list(default_nk_k_sequence(10))
    [2, 3, 4]

    >>> list(default_nk_k_sequence(20))
    [2, 3, 4, 5, 7, 9]

    >>> list(default_nk_k_sequence(32))
    [2, 4, 5, 8, 11, 15]

    >>> try:
    ...     default_nk_k_sequence(3)
    ... except ValueError as ve:
    ...     print(ve)
    n=3 is invalid, must be in 6..1000000000.

    >>> try:
    ...     default_nk_k_sequence("6")
    ... except TypeError as te:
    ...     print(te)
    n should be an instance of int but is str, namely '6'.
    """
    check_int_range(n, "n", 6)
    nhalf: Final[int] = n // 2
    ofs: Final[int] = nhalf - 2
    base: Final[set[int]] = {1 + (j * ofs) // 4 for j in range(5)}
    base.add(2)
    base.add(nhalf - 1)
    base.add(isqrt(n))
    return (i for i in sorted(base) if 1 < i < nhalf)


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

    @classmethod
    def default_instances(
            cls: type, scale_min: int = 6, scale_max: int = 32) \
            -> Iterator[Callable[[], "BitStringNKProblem"]]:
        """
        Get the default instances of this problem type.

        :return: an :class:`Iterator` with the instances
        """
        return (cast(Callable[[], "BitStringNKProblem"],
                     lambda __i=i, __j=j: cls(__i, __j))
                for i in default_scale_sequence(scale_min, scale_max)
                for j in default_nk_k_sequence(i))
