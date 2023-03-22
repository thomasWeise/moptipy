"""An implementation of an integer string based search space."""

from typing import Final

import numpy as np

from moptipy.spaces.nparrayspace import NPArraySpace
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import int_range_to_dtype
from moptipy.utils.strings import num_to_str_for_name
from moptipy.utils.types import type_error

#: the log key for the minimum value
KEY_MIN: Final[str] = "min"
#: the log key for the maximum value
KEY_MAX: Final[str] = "max"


class IntSpace(NPArraySpace):
    """
    A space where each element is a one-dimensional numpy integer array.

    Such spaces can serve as basis for implementing combinatorial
    optimization and can be extended to host permutations. Their elements are
    instances of :class:`numpy.ndarray`.

    >>> s = IntSpace(5, -4, 99)
    >>> print(s.dimension)
    5
    >>> print(s.min_value)
    -4
    >>> print(s.max_value)
    99
    >>> print(s.dtype)
    int8
    >>> s = IntSpace(5, 2, 200)
    >>> print(s.dtype)
    uint8
    >>> s = IntSpace(5, 2, 202340)
    >>> print(s.dtype)
    int32
    """

    def __init__(self, dimension: int,
                 min_value: int, max_value: int) -> None:
        """
        Create the integer-based search space.

        :param dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param min_value: the minimum value
        :param max_value: the maximum value
        """
        if not isinstance(min_value, int):
            raise type_error(min_value, "min_value", int)
        if not isinstance(max_value, int):
            raise type_error(max_value, "max_value", int)
        if min_value >= max_value:
            raise ValueError(
                f"max_value > min_value must hold, but got "
                f"min_value={min_value} and max_value={max_value}.")
        super().__init__(dimension, int_range_to_dtype(
            min_value=min_value, max_value=max_value))
        #: the lower bound, i.e., the minimum permitted value
        self.min_value: Final[int] = min_value
        #: the upper bound, i.e., the maximum permitted value
        self.max_value: Final[int] = max_value

    def create(self) -> np.ndarray:
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
        return np.full(shape=self.dimension, fill_value=self.min_value,
                       dtype=self.dtype)

    def validate(self, x: np.ndarray) -> None:
        """
        Validate an integer string.

        :param x: the integer string
        :raises TypeError: if the string is not an :class:`numpy.ndarray`.
        :raises ValueError: if the shape or data type of the vector is wrong
            or any of its element is not finite or if an element is out of
            the bounds.
        """
        super().validate(x)

        miv: Final[int] = self.min_value
        mav: Final[int] = self.max_value
        for index, item in enumerate(x):
            if not (miv <= item <= mav):
                raise ValueError(
                    f"x[{index}]={item}, but should be in {miv}..{mav}.")

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
        return (f"ints{self.dimension}{self.dtype.char}"
                f"{num_to_str_for_name(self.min_value)}to"
                f"{num_to_str_for_name(self.max_value)}")

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> space = IntSpace(7, -5, 5)
        >>> space.dimension
        7
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         space.log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[-2]
        'max: 5'
        >>> text[-3]
        'min: -5'
        >>> text[-4]
        'dtype: b'
        >>> text[-5]
        'nvars: 7'
        >>> text[-6]
        'class: moptipy.spaces.intspace.IntSpace'
        >>> text[-7]
        'name: ints7bm5to5'
        >>> len(text)
        8
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_MIN, self.min_value)
        logger.key_value(KEY_MAX, self.max_value)
