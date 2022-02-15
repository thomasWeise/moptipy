"""An implementation of a bit string based search space."""
from typing import Final

import numpy as np

from moptipy.spaces._nparrayspace import _NPArraySpace
from moptipy.utils.types import bool_to_str, str_to_bool

#: the internal type for but strings
_DTYPE: Final[np.dtype] = np.dtype(np.bool_)


class BitStrings(_NPArraySpace):
    """
    A space where each element is a bit string.

    With such a space, discrete optimization can be realized.
    """

    def __init__(self, dimension: int) -> None:
        """
        Create the vector-based search space.

        :param int dimension: The dimension of the search space,
            i.e., the number of decision variables.
        """
        super().__init__(dimension, _DTYPE)

    def create(self) -> np.ndarray:
        """
        Create a bit string filled with `False`.

        :return: the string
        :rtype: np.ndarray

        >>> from moptipy.spaces.bitstrings import BitStrings
        >>> s = BitStrings(8)
        >>> v = s.create()
        >>> print(s.to_str(v))
        FFFFFFFF
        >>> print(v.dtype)
        bool
        """
        return np.zeros(shape=self.dimension, dtype=_DTYPE)

    def to_str(self, x: np.ndarray) -> str:
        """
        Convert a bit string to a string, using `T` and `F` without separator.

        :param np.ndarray x: the bit string
        :return: the string
        :rtype: str
        """
        return "".join([bool_to_str(xx) for xx in x])

    def from_str(self, text: str) -> np.ndarray:
        """
        Convert a string to a bit string.

        :param str text: the text
        :return: the vector
        :raises TypeError: if `text` is not a `str`
        :raises ValueError: if `text` cannot be converted to a valid vector
        """
        if not (isinstance(text, str)):
            raise TypeError(f"text must be str, but is {type(text)}.")
        x: Final[np.ndarray] = self.create()
        x[:] = [str_to_bool(t) for t in text]
        self.validate(x)
        return x

    def n_points(self) -> int:
        """
        Get the scale of the bit string space.

        :return: 2 ** dimension
        :rtype: int

        >>> print(BitStrings(4).n_points())
        16
        """
        return 1 << self.dimension  # = 2 ** self.dimension

    def get_name(self) -> str:
        """
        Get the name of this space.

        :return: "bits" + dimension
        :rtype: str

        >>> print(BitStrings(5).get_name())
        bits5
        """
        return f"bits{self.dimension}"
