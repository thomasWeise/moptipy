"""An implementation of an integer string based search space."""
from typing import Final

import numpy as np

from moptipy.spaces._nparrayspace import _NPArraySpace
from moptipy.utils.logger import KeyValueSection
from moptipy.utils.nputils import int_range_to_dtype

#: the minimum value
KEY_MIN: Final[str] = "min"
#: the maximum value
KEY_MAX: Final[str] = "max"


class IntSpace(_NPArraySpace):
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

        :param int dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param int min_value: the minimum value
        :param int max_value: the maximum value
        """
        super().__init__(
            dimension,
            int_range_to_dtype(min_value=min_value, max_value=max_value))
        #: The minimum permitted value.
        self.min_value: Final[int] = min_value
        #: The maximum permitted value.
        self.max_value: Final[int] = max_value

    def create(self) -> np.ndarray:
        """
        Create a integer vector filled with the minimal value.

        :return: the vector
        :rtype: np.ndarray

        >>> from moptipy.spaces.intspace import IntSpace
        >>> s = IntSpace(dimension=12, min_value=5, max_value=332)
        >>> v = s.create()
        >>> print(s.to_str(v))
        5,5,5,5,5,5,5,5,5,5,5,5
        >>> print(v.dtype)
        int16
        """
        return np.full(shape=self.dimension,
                       fill_value=self.min_value,
                       dtype=self.dtype)

    def from_str(self, text: str) -> np.ndarray:
        """
        Convert a string to an integer string.

        :param str text: the text
        :return: the vector
        :rtype: np.ndarray
        :raises TypeError: if `text` is not a `str`
        :raises ValueError: if `text` cannot be converted to a valid vector
        """
        if not (isinstance(text, str)):
            raise TypeError(f"text must be str, but is {type(text)}.")
        x = np.fromstring(text, dtype=self.dtype, sep=",")
        self.validate(x)
        return x

    def validate(self, x: np.ndarray) -> None:
        """
        Validate an integer string.

        :param np.ndarray x: the integer string
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
        :rtype: int

        >>> print(IntSpace(4, -1, 3).n_points())
        625
        """
        return (self.max_value - self.min_value + 1) ** self.dimension

    def __str__(self) -> str:
        """
        Get the name of this integer space.

        :return: "ints" + dimension + dtype.char + min_value + "-" + max_value
        :rtype: int

        >>> print(IntSpace(4, -1, 3))
        ints4b-1-3
        """
        return f"ints{self.dimension}{self.dtype.char}" \
               f"{self.min_value}-{self.max_value}"

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param KeyValueLogger logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_MIN, self.min_value)
        logger.key_value(KEY_MAX, self.max_value)
