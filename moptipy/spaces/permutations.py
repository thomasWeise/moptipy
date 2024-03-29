"""
An implementation of a search space for permutations of a base string.

The base string can be a string where each element occurs exactly once,
e.g., `[0, 1, 2, 3]` or `[0, 7, 2, 5]`. It could also be a string where
some or all elements occur multiple times, e.g., a permutation with
repetition, such as `[0, 7, 7, 4, 5, 3, 3, 3]` or `[0, 0, 1, 1, 2, 2]`.
Search operators working on elements of this space are given in module
:mod:`moptipy.operators.permutations`.
"""
from math import factorial
from typing import Counter, Final, Iterable

import numpy as np
from pycommons.types import check_int_range, type_error

from moptipy.spaces.intspace import IntSpace
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import array_to_str
from moptipy.utils.strings import sanitize_name

#: the base string to be permuted
KEY_BASE_STRING: Final[str] = "baseString"
#: the number of times each value must occur
KEY_REPETITIONS: Final[str] = "repetitions"


class Permutations(IntSpace):  # +book
    """
    A space of permutations of a base string stored as :class:`numpy.ndarray`.

    This class includes standard permutations of the form 0, 1, 2, ..., n-1,
    but also permutations with repetitions.
    The idea is that a base string is defined, i.e., an array of integer
    values. In this array, some values may appear twice, some may be missing.
    For example, `[1, 3, 5, 5, 7]` is a proper base string. The space defined
    over this base string would then contain values such as `[1, 3, 5, 5, 7]`,
    `[1, 5, 3, 5, 7]`, `[7, 5, 5, 3, 1]` and so on. Basically, it will contain
    all strings that can be created by shuffling the base string. These
    strings have in common that they contain exactly all the values from the
    base string and contain them exactly as same as often as they appear in
    the base string. The space defined upon the above base string therefore
    would contain 5! / (1! * 1! * 2! * 1!) = 120 / 2 = 60 different strings.

    The permutation space defined above can be created as follows:

    >>> perm = Permutations([1, 3, 5, 5, 7])
    >>> print(perm.to_str(perm.blueprint))
    1;3;5;5;7
    >>> print(perm)
    permOfString
    >>> print(perm.n_points())
    60

    Another example is this:

    >>> perm = Permutations((1, 2, 3, 3, 2))
    >>> print(perm.to_str(perm.blueprint))
    1;2;2;3;3
    >>> print(perm)
    permOfString
    >>> print(perm.n_points())
    30

    If you want to create a permutation with repetitions, e.g.,
    where each of the n=4 values from 0 to 3 appear exactly 3 times,
    you can use the utility method `with_repetitions`:

    >>> perm = Permutations.with_repetitions(4, 3)
    >>> print(perm.to_str(perm.blueprint))
    0;0;0;1;1;1;2;2;2;3;3;3
    >>> print(perm)
    perm4w3r
    >>> print(perm.n_points())
    369600

    If you instead want to create standard permutations, i.e., where
    each value from 0 to n-1 appears exactly once, you would do:

    >>> perm = Permutations.standard(5)
    >>> print(perm.to_str(perm.blueprint))
    0;1;2;3;4
    >>> print(perm)
    perm5
    >>> print(perm.n_points())
    120
    """

    def __init__(self, base_string: Iterable[int]) -> None:  # +book
        """
        Create the space of permutations of a base string.

        :param base_string: an iterable of integer to denote the base string
        :raises TypeError: if one of the parameters has the wrong type
        :raises ValueError: if the parameters have the wrong value
        """
        if not isinstance(base_string, Iterable):
            raise type_error(base_string, "base_string", Iterable)

        string: Final[list[int]] = sorted(base_string)
        total: Final[int] = len(string)
        if total <= 0:
            raise ValueError(
                f"base string must not be empty, but is {base_string}.")

        # get data ranges
        self.__shape: Final[Counter[int]] = Counter[int]()
        minimum: int = string[0]
        maximum: int = string[0]
        for i in string:
            if not isinstance(i, int):
                raise type_error(i, "element of base_string", int)
            self.__shape[i] += 1
            minimum = min(minimum, i)
            maximum = max(maximum, i)

        # checking that the data is not empty
        different: Final[int] = len(self.__shape)
        if different <= 1:
            raise ValueError(
                "base_string must contain at least two different "
                f"elements, but is {base_string}.")

        # creating super class
        super().__init__(total, minimum, maximum)

        # start book
        #: a numpy array of the right type with the base string
        self.blueprint: Final[np.ndarray] = \
            np.array(string, dtype=self.dtype)
        # end book

        npoints: int = factorial(total)
        rep: int | None = self.__shape.get(minimum)
        for v in self.__shape.values():
            npoints = npoints // factorial(v)
            if v != rep:
                rep = None
        #: the exact number of different permutations
        self.__n_points = npoints

        #: the number of repetitions if all elements occur as same
        #: as often, or None otherwise
        self.__repetitions: Final[int | None] = rep

    def has_repetitions(self) -> bool:
        """
        Return whether elements occur repeatedly in the base string.

        :returns: `True` if at least one element occurs more than once,
            `False` otherwise
        """
        return (self.__repetitions is None) or (self.__repetitions > 1)

    def n(self) -> int:
        """
        Get the number of different values in the base string.

        :returns: the number of different values in the base string
        """
        return len(self.__shape)

    def is_dense(self) -> bool:
        """
        Check if all values in `min..max` appear in the permutation.

        :returns: `True` if the permutation is dense in the sense that
            all values from `self.min_value` to `self.max_value` appear.
        """
        return len(self.__shape) == (self.max_value - self.min_value + 1)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)

        reps: Final[int | None] = self.__repetitions
        if reps:
            logger.key_value(KEY_REPETITIONS, reps)
            if self.is_dense():
                return
        logger.key_value(KEY_BASE_STRING,
                         array_to_str(self.blueprint))

    def create(self) -> np.ndarray:  # +book
        r"""
        Create a permutation equal to the base string.

        The result is of the form [0, 0, 1, 1, 1, 2, 2...].

        :return: the permutation of the base string

        >>> perm = Permutations([1, 5, 2, 2, 4, 3, 4])
        >>> x = perm.create()
        >>> print(perm.to_str(x))
        1;2;2;3;4;4;5
        """
        return self.blueprint.copy()  # Create copy of the blueprint. # +book

    def validate(self, x: np.ndarray) -> None:
        """
        Validate a permutation of the base string.

        :param x: the integer string
        :raises TypeError: if the string is not an :class:`numpy.ndarray`.
        :raises ValueError: if the element is not a valid element of this
            permutation space, e.g., if the shape or data type of `x` is
            wrong, if an element is outside of the bounds, or if an
            element does not occur exactly the prescribed amount of times
        """
        super().validate(x)
        counts: Counter[int] = Counter[int]()
        for xx in x:
            counts[xx] += 1

        if counts != self.__shape:
            for key in sorted(set(counts.keys()).union(
                    set(self.__shape.keys()))):
                exp = self.__shape.get(key, 0)
                found = counts.get(key, 0)
                if found != exp:
                    raise ValueError(
                        f"Expected to find {key} exactly {exp} times "
                        f"but found it {found} times.")

    def n_points(self) -> int:
        """
        Get the number of possible different permutations of the base string.

        :return: factorial(dimension) / Product(factorial(count(e)) for all e)

        >>> print(Permutations([0, 1, 2, 3, 0, 1, 2, 3]).n_points())
        2520
        """
        return self.__n_points

    def __str__(self) -> str:
        """
        Get the name of this space.

        :return: "perm" + blueprint string

        >>> print(Permutations([0, 1, 0, 2, 1]))
        permOfString
        >>> print(Permutations([0, 2, 0, 1, 1, 2]))
        perm3w2r
        >>> print(Permutations([0, 2, 1, 3]))
        perm4
        >>> print(Permutations([3, 2, 4, 2, 3, 2,4, 3, 4]))
        perm2to4w3r
        """
        minimum: Final[int] = self.min_value
        maximum: Final[int] = self.max_value
        reps: Final[int | None] = self.__repetitions
        different: Final[int] = self.n()
        if reps and (different != (self.dimension // reps)):
            raise ValueError(f"huh? {different} != {self.dimension} / {reps}")
        all_values: Final[bool] = self.is_dense()
        min_is_0: Final[bool] = minimum == 0

        if reps:
            if reps == 1:
                if all_values:
                    if min_is_0:
                        return f"perm{different}"
                    return sanitize_name(f"perm{minimum}to{maximum}")
            elif all_values:  # repetitions != 1
                if min_is_0:
                    return f"perm{different}w{reps}r"
                return sanitize_name(f"perm{minimum}to{maximum}w{reps}r")
        return "permOfString"

    @staticmethod
    def standard(n: int) -> "Permutations":
        """
        Create a space for permutations of 0..n-1.

        :param n: the range of the values
        :returns: the permutations space
        """
        return Permutations(range(check_int_range(n, "n", 2, 100_000_000)))

    @staticmethod  # +book
    def with_repetitions(n: int, repetitions: int) -> "Permutations":  # +book
        """
        Create a space for permutations of `0..n-1` with repetitions.

        :param n: the range of the values
        :param repetitions: how often each value occurs
        :returns: the permutations space
        """
        check_int_range(repetitions, "repetitions", 1, 1_000_000)
        if repetitions <= 1:
            return Permutations.standard(n)
        check_int_range(n, "n", 1, 100_000_000)
        return Permutations(list(range(n)) * repetitions)  # +book
