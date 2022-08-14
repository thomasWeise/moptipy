"""An implementation of an integer string based search space."""
from typing import Final

import numpy

from moptipy.spaces.nparrayspace import NPArraySpace
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import int_range_to_dtype
from moptipy.utils.strings import sanitize_name

#: the minimum value
KEY_MIN: Final[str] = "min"
#: the maximum value
KEY_MAX: Final[str] = "max"


class IntSpace(NPArraySpace):
    """
    A space where each element is a one-dimensional numpy integer array.

    Such spaces can serve as basis for implementing combinatorial
    optimization and can be extended to host permutations.
    """

    def __init__(self, dimension: int,
                 min_value: int,
                 max_value: int) -> None:
        """
        Create the integer-based search space.

        :param dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param min_value: the minimum value
        :param max_value: the maximum value
        """
        super().__init__(
            dimension,
            int_range_to_dtype(min_value=min_value, max_value=max_value))
        #: The minimum permitted value.
        self.min_value: Final[int] = min_value
        #: The maximum permitted value.
        self.max_value: Final[int] = max_value

    def create(self) -> numpy.ndarray:
        """
        Create an integer vector filled with the minimal value.

        :return: the vector

        >>> from moptipy.spaces.intspace import IntSpace
        >>> s = IntSpace(dimension=12, min_value=5, max_value=332)
        >>> v = s.create()
        >>> print(s.to_str(v))
        5;5;5;5;5;5;5;5;5;5;5;5
        >>> print(v.dtype)
        int16
        """
        return numpy.full(shape=self.dimension,
                          fill_value=self.min_value,
                          dtype=self.dtype)

    def validate(self, x: numpy.ndarray) -> None:
        """
        Validate an integer string.

        :param x: the integer string
        :raises TypeError: if the string is not an element of this space.
        :raises ValueError: if the shape of the vector is wrong or any of its
            element is not finite.
        """
        super().validate(x)
        if not all(x >= self.min_value):
            raise ValueError(
                f"All elements of x must be >= {self.min_value}, but "
                f"{x.min()} was encountered.")
        if not all(x <= self.max_value):
            raise ValueError(
                f"All elements of x must be <= {self.max_value}, but "
                f"{x.max()} was encountered.")

    def n_points(self) -> int:
        """
        Get the number of possible different integer strings.

        :return: (max_value - min_value + 1) ** dimension

        >>> print(IntSpace(4, -1, 3).n_points())
        625
        """
        return (self.max_value - self.min_value + 1) ** self.dimension

    def __str__(self) -> str:
        """
        Get the name of this integer space.

        :return: "ints" + dimension + dtype.char + min_value + "-" + max_value

        >>> print(IntSpace(4, -1, 3))
        ints4bm1to3
        """
        return sanitize_name(f"ints{self.dimension}{self.dtype.char}"
                             f"{self.min_value}to{self.max_value}")

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_MIN, self.min_value)
        logger.key_value(KEY_MAX, self.max_value)
