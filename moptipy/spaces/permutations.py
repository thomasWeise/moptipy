"""An implementation of a search space for permutations."""
from math import factorial

import numpy as np

from moptipy.spaces.intspace import IntSpace


class Permutations(IntSpace):
    """A space of permutations represented as 1-dimensional int arrays."""

    def __init__(self, n: int) -> None:
        """
        Create the space of permutations of n elements.

        :param int n: the length of the permutations
        """
        super().__init__(dimension=n, min_value=0, max_value=n - 1)

        self.__blueprint = super().create()
        self.__blueprint[0:n] = list(range(n))

    def create(self) -> np.ndarray:
        """
        Create a permutation of the form [0, 1, 2, 4, 5, ...].

        :return: the permutation
        :rtype: np.ndarray

        >>> from moptipy.spaces.permutations import Permutations
        >>> p = Permutations(12)
        >>> perm = p.create()
        >>> print(p.to_str(perm))
        0,1,2,3,4,5,6,7,8,9,10,11
        """
        return self.__blueprint.copy()

    def scale(self) -> int:
        """
        Get the number of different permutations.

        :return: factorial(dimension)
        :rtype: int
        """
        return factorial(self.dimension)

    def validate(self, x: np.ndarray) -> None:
        """
        Validate a permutation.

        :param np.ndarray x: the integer string
        :raises TypeError: if the string is not an element of this space.
        :raises ValueError: if the shape of the vector is wrong or any of its
            element is not finite.
        """
        super().validate(x)
        counts = np.zeros(self.dimension, np.dtype(np.int32))
        for xx in x:
            counts[xx] += 1
        if any(counts != 1):
            raise ValueError(
                f"Each element in 0..{self.dimension - 1} must occur exactly "
                f"once, but encountered {super().to_str(counts[counts != 1])} "
                "occurrences.")

    def get_name(self) -> str:
        """
        Get the name of this permutation space.

        :return: "perm" + dimension
        :rtype: str
        """
        return f"perm{self.dimension}"
