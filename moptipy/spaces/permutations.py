"""An implementation of a search space for permutations of a base string."""
from math import factorial
from typing import Final, Iterable, Dict, Optional, List

import numpy as np

from moptipy.spaces.intspace import IntSpace
from moptipy.utils.logger import KeyValueSection

#: the base string to be permuted
KEY_BASE_STRING: Final[str] = "baseString"
#: the number of times each value must occur
KEY_REPETITIONS: Final[str] = "repetitions"


class Permutations(IntSpace):  # +book
    """
    A space of permutations of a base string.

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
    1,3,5,5,7
    >>> print(perm)
    perm_of_string
    >>> print(perm.n_points())
    60

    Another example is this:

    >>> perm = Permutations((1, 2, 3, 3, 2))
    >>> print(perm.to_str(perm.blueprint))
    1,2,2,3,3
    >>> print(perm)
    perm_of_string
    >>> print(perm.n_points())
    30

    If you want to create a permutation with repetitions, e.g.,
    where each of the n=4 values from 0 to 3 appear exactly 3 times,
    you can use the utility method `with_repetitions`:

    >>> perm = Permutations.with_repetitions(4, 3)
    >>> print(perm.to_str(perm.blueprint))
    0,0,0,1,1,1,2,2,2,3,3,3
    >>> print(perm)
    perm4w3r
    >>> print(perm.n_points())
    369600

    If you instead want to create standard permutations, i.e., where
    each value from 0 to n-1 appears exactly once, you would do:

    >>> perm = Permutations.standard(5)
    >>> print(perm.to_str(perm.blueprint))
    0,1,2,3,4
    >>> print(perm)
    perm5
    >>> print(perm.n_points())
    120
    """

    def __init__(self, base_string: Iterable[int]) -> None:  # +book
        """
        Create the space of permutations of a base string.

        :param Iterable[int] base_string: an iterable of integer to denote
            the base string
        :raises TypeError: if one of the parameters has the wrong type
        :raises ValueError: if the parameters have the wrong value
        """
        if not isinstance(base_string, Iterable):
            raise TypeError(
                f"base_string must be iterable, but is {type(base_string)}.")

        string: Final[List[int]] = sorted(base_string)
        total: Final[int] = len(string)
        if total <= 0:
            raise ValueError(
                f"base string must not be empty, but is {base_string}.")

        # get data ranges
        self.__shape: Final[Dict[int, int]] = {}
        minimum: int = string[0]
        maximum: int = string[0]
        for i in string:
            if not isinstance(i, int):
                raise TypeError(f"base_string must only contain int, but "
                                f"contains {i}, which is {type(i)}.")
            self.__shape[i] = self.__shape.get(i, 0) + 1
            if i < minimum:
                minimum = i
            if i > maximum:
                maximum = i

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
        rep: Optional[int] = self.__shape.get(minimum)
        for v in self.__shape.values():
            npoints = npoints // factorial(v)
            if v != rep:
                rep = None
        #: the exact number of different permutations
        self.__n_points = npoints

        #: the number of repetitions if all elements occur as same
        #: as often, or None otherwise
        self.__repetitions: Final[Optional[int]] = rep

    def has_repetitions(self) -> bool:
        """
        Return whether elements occur repeatedly in the base string.

        :returns: `True` if at least one element occurs more than once,
            `False` otherwise
        :rtype: bool
        """
        return (self.__repetitions is None) or (self.__repetitions > 1)

    def n(self) -> int:
        """
        Get the number of different values in the base string.

        :returns: the number of different values in the base string
        :rtype: int
        """
        return len(self.__shape)

    def is_dense(self) -> bool:
        """
        Check if all values in `min..max` appear in the permutation.

        :returns: `True` if the permutation is dense in the sense that
            all values from `self.min_value` to `self.max_value` appear.
        :rtype: bool
        """
        return len(self.__shape) == (self.max_value - self.min_value + 1)

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param KeyValueLogger logger: the logger
        """
        super().log_parameters_to(logger)

        reps: Final[Optional[int]] = self.__repetitions
        if reps:
            logger.key_value(KEY_REPETITIONS, reps)
            if self.is_dense():
                return
        logger.key_value(KEY_BASE_STRING,
                         ",".join([str(xx) for xx in self.blueprint]))

    def create(self) -> np.ndarray:  # +book
        r"""
        Create a permutation equal to the base string.

        The result is of the form [0, 0, 1, 1, 1, 2, 2...].

        :return: the permutation of the base string
        :rtype: np.ndarray

        >>> perm = Permutations([1, 5, 2, 2, 4, 3, 4])
        >>> x = perm.create()
        >>> print(perm.to_str(x))
        1,2,2,3,4,4,5
        """
        return self.blueprint.copy()  # +book

    def validate(self, x: np.ndarray) -> None:
        """
        Validate a permutation of the base string.

        :param np.ndarray x: the integer string
        :raises TypeError: if the string is not an element of this space.
        :raises ValueError: if the shape of the vector is wrong or any of its
            element is not finite.
        """
        super().validate(x)
        counts: Dict[int, int] = {}
        for xx in x:
            counts[xx] = counts.get(xx, 0) + 1

        if counts != self.__shape:
            for key in sorted(set(counts.keys()).union(
                    set(self.__shape.keys()))):
                exp = self.__shape.get(key, 0)
                found = counts.get(key, 0)
                if found != exp:
                    raise ValueError(
                        f"expected to find {key} exactly {exp} times "
                        f"but found it {found} times.")

    def n_points(self) -> int:
        """
        Get the number of possible different permutations of the base string.

        :return: factorial(simension) / Product(factorial(count(e)) for all e)
        :rtype: int

        >>> print(Permutations([0, 1, 2, 3, 0, 1, 2, 3]).n_points())
        2520
        """
        return self.__n_points

    def __str__(self) -> str:
        """
        Get the name of this space.

        :return: "perm" + blueprint string
        :rtype: str

        >>> print(Permutations([0, 1, 0, 2, 1]))
        perm_of_string
        >>> print(Permutations([0, 2, 0, 1, 1, 2]))
        perm3w2r
        >>> print(Permutations([0, 2, 1, 3]))
        perm4
        >>> print(Permutations([3, 2, 4, 2, 3, 2,4, 3, 4]))
        perm2to4w3r
        """
        minimum: Final[int] = self.min_value
        maximum: Final[int] = self.max_value
        reps: Final[Optional[int]] = self.__repetitions
        different: Final[int] = self.n()
        if reps and (different != (self.dimension // reps)):
            raise ValueError(f"huh? {different} != {self.dimension} / {reps}")
        all_values: Final[bool] = self.is_dense()
        min_is_0: Final[bool] = minimum <= 0

        if reps:
            if reps == 1:
                if all_values:
                    if min_is_0:
                        return f"perm{different}"
                    return f"perm{minimum}to{maximum}"
            else:  # repetitions != 1
                if all_values:
                    if min_is_0:
                        return f"perm{different}w{reps}r"
                    return f"perm{minimum}to{maximum}w{reps}r"
        return "perm_of_string"

    @staticmethod
    def standard(n: int) -> 'Permutations':
        """
        Create a space for permutations of 0..n-1.

        :param int n: the range of the values
        :returns: the permutations space
        :rtype: Permutations
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be int, but is {type(n)}.")
        if n <= 1:
            raise ValueError(f"n must be >1, but is {n}.")
        return Permutations(range(n))

    @staticmethod  # +book
    def with_repetitions(n: int, repetitions: int) -> 'Permutations':  # +book
        """
        Create a space for permutations of `0..n-1` with repetitions.

        :param int n: the range of the values
        :param int repetitions: how often each value occurs
        :returns: the permutations space
        :rtype: Permutations
        """
        if not isinstance(repetitions, int):
            raise TypeError(
                f"repetitions must be int, but is {type(repetitions)}.")
        if repetitions < 1:
            raise ValueError(f"repetitions must be >0, but is {repetitions}.")
        if repetitions <= 1:
            return Permutations.standard(n)

        if not isinstance(n, int):
            raise TypeError(f"n must be int, but is {type(n)}.")
        if n <= 1:
            raise ValueError(f"n must be >1, but is {n}.")
        return Permutations(list(range(n)) * repetitions)  # +book
