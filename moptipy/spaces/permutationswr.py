"""An implementation of a search space for permutations with repetitions."""
from math import factorial
from typing import Final

import numpy as np

from moptipy.spaces.intspace import IntSpace
from moptipy.utils.logger import KeyValueSection

#: the number of times each value must occur
KEY_REPETITIONS: Final[str] = "repetitions"


def permutations_with_repetitions_space_size(
        n: int, repetitions: int = 1) -> int:
    """
    Compute the number of different permutations with repetitions.

    :param int n: number of different values
    :param int repetitions: the number of repetitions
    :returns: the number of permutations with repetitions, i.e.,
        factorial(n*repetitions) / (factorial(repetitions) ** n)
    :rtype: int

     >>> print(permutations_with_repetitions_space_size(4, 3))
     369600
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be integer, but is {type(n)}.")
    if (n <= 1) or (n > 1_000_000_000):
        raise ValueError(f"n must be in 2..1_000_000_000, but is {n}.")
    if not isinstance(repetitions, int):
        raise TypeError(
            f"repetitions must be integer, but is {type(repetitions)}.")
    if (repetitions <= 0) or (repetitions > 1_000_000_000):
        raise ValueError("repetitions must be in 1..1_000_000_000, "
                         f"but is {repetitions}.")
    return factorial(n * repetitions) // (factorial(repetitions) ** n)


# start book
class PermutationsWithRepetitions(IntSpace):
    """A space of permutations with repetitions."""

    def __init__(self, n: int, repetitions: int = 1) -> None:
        # end book
        """
        Create the permutation-with-repetitions space.

        :param int n: number of different values
        :param int repetitions: the number of repetitions
        :raises TypeError: if one of the parameters has the wrong type
        :raises ValueError: if the parameters have the wrong value
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be integer, but is {type(n)}.")
        if (n <= 1) or (n > 1_000_000_000):
            raise ValueError(f"n must be in 2..1_000_000_000, but is {n}.")
        if not isinstance(repetitions, int):
            raise TypeError(
                f"repetitions must be integer, but is {type(repetitions)}.")
        if (repetitions <= 0) or (repetitions > 1_000_000_000):
            raise ValueError("repetitions must be in 1..1_000_000_000, "
                             f"but is {repetitions}.")

        super().__init__(dimension=n * repetitions, min_value=0,
                         max_value=n - 1)
        #: n is the number of items, meaning the values are in [0, n-1].
        self.n: Final[int] = n
        #: The number of times each value must occur.
        self.repetitions: Final[int] = repetitions

        # start book
        # ...omitted some things, self.dimension = n*repetitions
        #: the blueprint of a valid solution: a canonical permutation
        #: with repetitions
        self.blueprint: Final[np.ndarray] = np.empty(
            shape=self.dimension, dtype=self.dtype)
        self.blueprint[0:self.dimension] = list(range(n)) * repetitions
        # end book

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param KeyValueLogger logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_REPETITIONS, self.repetitions)

    def create(self) -> np.ndarray:  # +book
        r"""
        Create a permutation with repetitions.

        The result is of the form [0, 1, 2, ..., 0, 1, 2...].

        :return: the permutation with repetitions
        :rtype: np.ndarray

        >>> pwr = PermutationsWithRepetitions(4, 3)
        >>> perm = pwr.create()
        >>> print(pwr.to_str(perm))
        0,1,2,3,0,1,2,3,0,1,2,3
        """
        return self.blueprint.copy()  # +book

    def validate(self, x: np.ndarray) -> None:
        """
        Validate a permutation with repetitions.

        :param np.ndarray x: the integer string
        :raises TypeError: if the string is not an element of this space.
        :raises ValueError: if the shape of the vector is wrong or any of its
            element is not finite.
        """
        super().validate(x)
        counts = np.zeros(self.n, np.dtype(np.int32))
        for xx in x:
            counts[xx] += 1
        if any(counts != self.repetitions):
            raise ValueError(
                f"Each element in 0..{self.n - 1} must occur exactly "
                f"{self.repetitions} times, but encountered "
                f"{super().to_str(counts[counts != self.repetitions])}"
                " occurrences.")

    def n_points(self) -> int:
        """
        Get the number of possible different permutations with repetitions.

        :return: factorial(n*repetitions) / (factorial(repetitions) ** n)
        :rtype: int

        >>> print(PermutationsWithRepetitions(4, 2).n_points())
        2520
        """
        return permutations_with_repetitions_space_size(
            self.n, self.repetitions)

    def __str__(self) -> str:
        """
        Get the name of this space.

        :return: "perm" + n + "w" + repetitions + "r"
        :rtype: str

        >>> print(PermutationsWithRepetitions(4, 3))
        perm4w3r
        """
        return f"perm{self.n}w{self.repetitions}r"
