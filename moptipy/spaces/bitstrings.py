"""An implementation of a bit string based search space."""
from typing import Final

import numpy as np
from pycommons.io.csv import CSV_SEPARATOR
from pycommons.strings.string_conv import bool_to_str, str_to_bool
from pycommons.types import type_error

from moptipy.spaces.nparrayspace import NPArraySpace
from moptipy.utils.nputils import DEFAULT_BOOL


class BitStrings(NPArraySpace):
    """
    A space where each element is a bit string (:class:`numpy.ndarray`).

    With such a space, discrete optimization can be realized.

    >>> s = BitStrings(5)
    >>> print(s.dimension)
    5
    >>> print(s.dtype)
    bool
    >>> print(s.create())
    [False False False False False]
    >>> print(s.to_str(s.create()))
    FFFFF
    >>> print(s.from_str(s.to_str(s.create())))
    [False False False False False]
    """

    def __init__(self, dimension: int) -> None:
        """
        Create the vector-based search space.

        :param dimension: The dimension of the search space,
            i.e., the number of decision variables.
        """
        super().__init__(dimension, DEFAULT_BOOL)

    def create(self) -> np.ndarray:
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
        return np.zeros(shape=self.dimension, dtype=DEFAULT_BOOL)

    def from_str(self, text: str) -> np.ndarray:
        """
        Convert a string to a bit string.

        :param text: the text
        :return: the vector
        :raises TypeError: if `text` is not a `str`
        :raises ValueError: if `text` cannot be converted to a valid vector
        """
        if not (isinstance(text, str)):
            raise type_error(text, "text", str)
        x: Final[np.ndarray] = self.create()
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

    def to_hex_string(self, x: np.ndarray) -> str:
        """
        Convert a bit string to a hexadecimal number.

        The reverse operator is given as :meth:`from_hex_string`.

        :param x: the bit string
        :returns: the hexadecimal number

        >>> b = BitStrings(134)
        >>> xt = b.create()
        >>> xt[:] = [((i % 3) == 0) != ((i % 7) == 0)
        ...         for i in range(b.dimension)]
        >>> "".join(reversed(b.to_str(xt)))
        'TTFFTFFFFFTFFTTFTFFTFTTFFTFFFFFTFFTTFTFFTFTTFFTFFFFFTFFTTFTFFTFT\
TFFTFFFFFTFFTTFTFFTFTTFFTFFFFFTFFTTFTFFTFTTFFTFFFFFTFFTTFTFFTFTTFFTFFF'
        >>> hex(int("11001000001001101001011001000001001101001011001000"
        ...         "0010011010010110010000010011010010110010000010011010"
        ...         "01011001000001001101001011001000", 2))
        '0x3209a5904d2c826964134b209a5904d2c8'
        >>> b.to_hex_string(xt)
        '3209a5904d2c826964134b209a5904d2c8'
        """
        self.validate(x)
        number: int = 0
        adder: int = 1
        for i in range(self.dimension):
            if x[i]:
                number += adder
            adder *= 2
        return format(number, "x")

    def from_hex_string(self, x: np.ndarray, s: str) -> None:
        """
        Convert a hexadecimal number to a numpy bit string.

        The reverse operator is given as :meth:`to_hex_string`.

        :param x: the bit string destination
        :param s: the string to convert

        >>> b = BitStrings(134)
        >>> x1 = b.create()
        >>> x1[:] = [((i % 3) == 0) != ((i % 7) == 0)
        ...         for i in range(b.dimension)]
        >>> x2 = b.create()
        >>> b.from_hex_string(x2, b.to_hex_string(x1))
        >>> b.is_equal(x1, x2)
        True
        """
        number: int = int(str.strip(s), 16)  # enforce type error
        for i in range(self.dimension):
            x[i] = (number & 1) != 0
            number >>= 1
        self.validate(x)

    def to_rle_str(self, x: np.ndarray) -> str:
        """
        Convert a numpy bit string to a run-length encoded string.

        The first bit value is stored, followed by a colon (:), followed by the
        run lengths separated with semicolons.
        The reverse operator is given as :meth:`from_rle_str`.

        :param x: the bit string
        :returns: the run-length encoded string

        >>> b = BitStrings(134)
        >>> xt = b.create()
        >>> xt[:] = [((i % 3) == 0) != ((i % 7) == 0)
        ...         for i in range(b.dimension)]
        >>> b.to_rle_str(xt)
        'F:3;1;2;2;1;1;2;1;1;2;2;1;5;1;2;2;1;1;2;1;1;2;2;1;5;1;2;2;1;1;2;1;1;\
2;2;1;5;1;2;2;1;1;2;1;1;2;2;1;5;1;2;2;1;1;2;1;1;2;2;1;5;1;2;2;1;1;2;1;1;2;2;\
1;5;1;2;2'

        >>> b = BitStrings(5)
        >>> xt = b.create()
        >>> xt.fill(0)
        >>> b.to_rle_str(xt)
        'F:5'
        >>> xt[1] = 1
        >>> b.to_rle_str(xt)
        'F:1;1;3'

        >>> b = BitStrings(5)
        >>> xt = b.create()
        >>> xt.fill(1)
        >>> b.to_rle_str(xt)
        'T:5'
        >>> xt[-1] = 0
        >>> b.to_rle_str(xt)
        'T:4;1'
        >>> xt[-1] = 1
        >>> xt[-2] = 0
        >>> b.to_rle_str(xt)
        'T:3;1;1'
        """
        self.validate(x)
        cur: np.bool = x[0]
        sep: str = ":"
        rle: list[str] = [bool_to_str(bool(cur))]
        start: int = 0
        for i, bit in enumerate(x):
            if bit != cur:
                rle.append(f"{sep}{i - start}")
                sep = CSV_SEPARATOR
                cur = bit
                start = i
        rle.extend(f"{sep}{self.dimension - start}")
        return "".join(rle)

    def from_rle_str(self, x: np.ndarray, s: str) -> None:
        """
        Convert a run-length encoded string to a numpy bit string.

        The reverse operator is given as :meth:`to_rle_str`.

        :param s: string to convert
        :param x: the bit string destination

        >>> b = BitStrings(134)
        >>> x1 = b.create()
        >>> x1[:] = [((i % 3) == 0) != ((i % 7) == 0)
        ...         for i in range(b.dimension)]
        >>> x2 = b.create()
        >>> b.from_hex_string(x2, b.to_hex_string(x1))
        >>> all(x1 == x2)
        True
        >>> x1[5] = not x1[5]
        >>> b.from_hex_string(x2, b.to_hex_string(x1))
        >>> all(x1 == x2)
        True

        >>> b = BitStrings(5)
        >>> x1 = b.create()
        >>> x2 = b.create()
        >>> x1.fill(0)
        >>> b.from_hex_string(x2, b.to_hex_string(x1))
        >>> all(x1 == x2)
        True

        >>> x1[1] = 1
        >>> b.from_hex_string(x2, b.to_hex_string(x1))
        >>> all(x1 == x2)
        True

        >>> x1.fill(1)
        >>> b.from_hex_string(x2, b.to_hex_string(x1))
        >>> all(x1 == x2)
        True
        """
        s = str.strip(s)
        value: bool = str_to_bool(s[0])
        i: int = 0
        for rl in s[s.index(":") + 1:].split(CSV_SEPARATOR):
            for _ in range(int(rl)):
                x[i] = value
                i += 1
            value = not value
        self.validate(x)
