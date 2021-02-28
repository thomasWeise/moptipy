import numpy as np

from moptipy.spaces.intspace import IntSpace
from math import factorial


class Permutations(IntSpace):
    """
    A space where each element is a one-dimensional numpy integer array
    and represents a permutation.
    """

    def __init__(self, n: int):
        """
        Create the space of permutations of n elements
        :param int n: the length of the permutations
        """
        super().__init__(dimension=n, min_value=0, max_value=n - 1)

        self.__blueprint = super().create()
        self.__blueprint[0:n] = list(range(n))

    def create(self) -> np.ndarray:
        """
        This method creates a permutation of the form [0, 1, 2, 4, 5, ...]
        :return: the permutation

        >>> from moptipy.spaces.permutations import Permutations
        >>> p = Permutations(12)
        >>> perm = p.create()
        >>> print(p.to_str(perm))
        0,1,2,3,4,5,6,7,8,9,10,11
        """
        return self.__blueprint.copy()

    def scale(self) -> int:
        return factorial(self.dimension)

    def validate(self, x: np.ndarray):
        super().validate(x)
        counts = np.zeros(self.dimension, np.dtype(np.int32))
        for xx in x:
            counts[xx] += 1
        if any(counts != 1):
            raise ValueError("Each element in 0.." + str(self.dimension - 1)
                             + " must occur exactly once, but encountered "
                             + super().to_str(counts[counts != 1])
                             + " occurrences.")

    def get_name(self) -> str:
        return "perm" + str(self.dimension)
