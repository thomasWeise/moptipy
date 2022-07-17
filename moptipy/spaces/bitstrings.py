"""An implementation of a bit string based search space."""
from typing import Final

import numpy

from moptipy.spaces.nparrayspace import NPArraySpace
from moptipy.utils.strings import str_to_bool
from moptipy.utils.types import type_error

#: the internal type for but strings
_DTYPE: Final[numpy.dtype] = numpy.dtype(numpy.bool_)


class BitStrings(NPArraySpace):
    """
    A space where each element is a bit string.

    With such a space, discrete optimization can be realized.
    """

    def __init__(self, dimension: int) -> None:
        """
        Create the vector-based search space.

        :param dimension: The dimension of the search space,
            i.e., the number of decision variables.
        """
        super().__init__(dimension, _DTYPE)

    def create(self) -> numpy.ndarray:
        """
        Create a bit string filled with `False`.

        :return: the string

        >>> from moptipy.spaces.bitstrings import BitStrings
        >>> s = BitStrings(8)
        >>> v = s.create()
        >>> print(s.to_str(v))
        FFFFFFFF
        >>> print(v.dtype)
        bool
        """
        return numpy.zeros(shape=self.dimension, dtype=_DTYPE)

    def from_str(self, text: str) -> numpy.ndarray:
        """
        Convert a string to a bit string.

        :param text: the text
        :return: the vector
        :raises TypeError: if `text` is not a `str`
        :raises ValueError: if `text` cannot be converted to a valid vector
        """
        if not (isinstance(text, str)):
            raise type_error(text, "text", str)
        x: Final[numpy.ndarray] = self.create()
        x[:] = [str_to_bool(t) for t in text]
        self.validate(x)
        return x

    def n_points(self) -> int:
        """
        Get the scale of the bit string space.

        :return: 2 ** dimension

        >>> print(BitStrings(4).n_points())
        16
        """
        return 1 << self.dimension  # = 2 ** self.dimension

    def __str__(self) -> str:
        """
        Get the name of this space.

        :return: "bits" + dimension

        >>> print(BitStrings(5))
        bits5
        """
        return f"bits{self.dimension}"
