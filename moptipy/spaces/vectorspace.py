"""An implementation of an unconstrained n-dimensional continuous space."""

import numpy as np

from moptipy.spaces._nparrayspace import _NPArraySpace
from moptipy.utils.nputils import is_np_float, is_all_finite
from moptipy.utils.types import float_to_str


class VectorSpace(_NPArraySpace):
    """
    A vector space where each element is a one-dimensional numpy float array.

    Such spaces are useful for continuous optimization.
    """

    def __init__(self, dimension: int, dtype=np.dtype(np.float64)) -> None:
        """
        Create the vector-based search space.

        :param int dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param dtype: The basic data type of the vector space,
            i.e., the type of the decision variables
        """
        super().__init__(dimension, dtype)
        if not (isinstance(self.dtype, np.dtype) and is_np_float(self.dtype)):
            raise TypeError(f"Invalid data type {dtype}.")

    def to_str(self, x: np.ndarray) -> str:
        """
        Convert a vector to a string, using `,` as separator.

        :param np.ndarray x: the vector
        :return: the string
        :rtype: str
        """
        return ",".join([float_to_str(xx) for xx in x])

    def from_str(self, text: str) -> np.ndarray:
        """
        Convert a string to a vector.

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
        Validate a vector.

        :param np.ndarray x: the real vector
        :raises TypeError: if the vector is not an element of this space.
        :raises ValueError: if the shape of the vector is wrong or any of its
            element is not finite.
        """
        if not is_all_finite(x):
            raise ValueError("All elements must be finite.")

    def n_points(self) -> int:
        """
        Get the number of different floating point values in this space.

        :return: The space contains unrestricted floating point numbers, so we
            return the approximate number of finite floating point numbers.
        :rtype: int

        >>> print(VectorSpace(3, np.dtype(np.float64)).n_points())
        6267911251143764491534102180507836301813760039183993274367
        """
        if self.dtype.char == "e":
            exponent = 5
            mantissa = 10
        elif self.dtype.char == "f":
            exponent = 8
            mantissa = 23
        elif self.dtype == "d":
            exponent = 11
            mantissa = 52
        elif self.dtype == "g":
            exponent = 15
            mantissa = 112
        else:
            raise ValueError(f"Invalid dtype {self.dtype}.")

        base = 2 * ((2 ** exponent) - 1) * (2 ** mantissa) - 1
        return base ** self.dimension

    def get_name(self) -> str:
        """
        Get the name of this space.

        :return: "vector" + dimension + dtype.char
        :rtype: str

        >>> print(VectorSpace(3, np.dtype(np.float64)).get_name())
        vector3d
        """
        return f"vector{self.dimension}{self.dtype.char}"
