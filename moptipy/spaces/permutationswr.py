"""An implementation of a search space for permutations with repetitions."""
from math import factorial
from typing import Final

import numpy as np

from moptipy.spaces.intspace import IntSpace
from moptipy.utils.logger import KeyValueSection


class PermutationsWithRepetitions(IntSpace):
    """A space where each element represents a permutation with repetition."""

    #: the number of times each value must occur
    KEY_REPETITIONS: Final = "repetitions"

    def __init__(self, n: int, repetitions: int = 1) -> None:
        """
        Create the permutation-with-repetitions space.

        :param int n: number of different values
        :param int repetitions: the number of repetitions
        :raises TypeError: if one of the parameters has the wrong type
        :raises ValueError: if the parameters have the wrong value
        """
        if not isinstance(n, int):
            raise TypeError("n must be integer, but is '"
                            + str(type(n)) + "'.")
        if (n <= 0) or (n > 1_000_000_000):
            raise ValueError("n must be in 1..1_000_000_000, but is "
                             + str(n) + ".")
        if not isinstance(repetitions, int):
            raise TypeError("repetitions must be integer, but is '"
                            + str(type(repetitions)) + "'.")
        if (repetitions <= 0) or (repetitions > 1_000_000_000):
            raise ValueError("repetitions must be in 1..1_000_000_000, "
                             "but is " + str(repetitions) + ".")

        super().__init__(dimension=n * repetitions,
                         min_value=0,
                         max_value=n - 1)

        self.n = n
        """n is the number of items, meaning the values are in [0, n-1]."""
        self.repetitions = repetitions
        """The number of times each value must occur."""

        self.__blueprint = super().create()
        self.__blueprint[0:self.dimension] = list(range(n)) * repetitions

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param KeyValueLogger logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value(PermutationsWithRepetitions.KEY_REPETITIONS,
                         self.repetitions)

    def create(self) -> np.ndarray:
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
        return self.__blueprint.copy()

    def scale(self) -> int:
        """
        Get the number of possible different permutations with repetitions.

        :return: factorial(n*repetitions) / (factorial(repetitions) ** n)
        :rtype: int
        """
        return factorial(self.n * self.repetitions) // \
            (factorial(self.repetitions) ** self.n)

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
                "Each element in 0.." + str(self.n - 1) + " must occur "
                + str(self.repetitions) + " times, but encountered "
                + super().to_str(counts[counts != self.repetitions])
                + " occurrences.")

    def get_name(self) -> str:
        """
        Get the name of this space.

        :return: "perm" + n + "w" + repetitions + "r"
        :rtype: str
        """
        return ("perm" + str(self.n)) + "w" + (str(self.repetitions) + "r")
