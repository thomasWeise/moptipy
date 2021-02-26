from moptipy.spaces.intspace import IntSpace
from typing import Final
from moptipy.utils.logger import KeyValuesSection
import numpy as np


class PermutationsWithRepetitions(IntSpace):
    """
    A space where each element is a one-dimensional numpy integer array
    and represents a permutation with repetition.
    """

    #: the number of times each value must occur
    KEY_REPETITIONS: Final = "repetitions"

    def __init__(self, n: int, repetitions: int = 1):
        super().__init__(dimension=n * repetitions,
                         min_value=0,
                         max_value=n - 1)
        if not isinstance(n, int):
            raise ValueError("n must be integer, but is '"
                             + str(type(n)) + "'.")
        if n <= 0:
            raise ValueError("n must be > 0, but is " + str(n) + ".")
        if not isinstance(repetitions, int):
            raise ValueError("repetitions must be integer, but is '"
                             + str(type(repetitions)) + "'.")
        if repetitions <= 0:
            raise ValueError("repetitions must be > 0, but is "
                             + str(repetitions) + ".")

        self.n = n
        """n is the number of items, meaning the values are in [0, n-1]."""
        self.repetitions = repetitions
        """The number of times each value must occur."""

        self.__blueprint = super().create()
        self.__blueprint[0:self.dimension] = list(range(n)) * repetitions

    def log_parameters_to(self, logger: KeyValuesSection):
        super().log_parameters_to(logger)
        logger.key_value(PermutationsWithRepetitions.KEY_REPETITIONS,
                         self.repetitions)

    def create(self) -> np.ndarray:
        return self.__blueprint.copy()

    def validate(self, x: np.ndarray):
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
        return ("perm" + str(self.n)) + "w" + (str(self.repetitions) + "r")
