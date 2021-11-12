"""An implementation of an unconstrained n-dimensional continuous space."""
from typing import Final

import numpy as np

from moptipy.api.space import Space
from moptipy.utils.logger import KeyValueSection
from moptipy.utils.logging import float_to_str, KEY_SPACE_NUM_VARS
from moptipy.utils.nputils import is_np_float, is_all_finite

#: the character identifying the numpy data type backing the space
KEY_NUMPY_TYPE: Final = "dtype"


class VectorSpace(Space):
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
        if not isinstance(dimension, int):
            raise TypeError(
                f"dimension must be integer, but got {type(dimension)}.")
        if (dimension < 1) or (dimension > 1_000_000_000):
            raise ValueError("dimension must be in 1..1_000_000_000, "
                             f"but got {dimension}.")

        if not (isinstance(dtype, np.dtype) and is_np_float(dtype)):
            raise TypeError(f"Invalid data type {dtype}.")

        #: The dimension of the space, i.e., the vectors.
        self.dimension: Final[int] = dimension

        #: The basic data type of the vector elements.
        self.dtype: Final[np.dtype] = dtype

    def create(self) -> np.ndarray:
        """
        Create a real vector filled with `0`.

        :return: the zero vector
        :rtype: np.ndarray
        """
        return np.zeros(shape=self.dimension, dtype=self.dtype)

    def copy(self, source: np.ndarray, dest: np.ndarray) -> None:
        """
        Copy the contents of one vector to another.

        :param np.ndarray source: the source vector
        :param np.ndarray dest: the target vector
        """
        np.copyto(dest, source)

    def to_str(self, x: np.ndarray) -> str:
        """
        Convert a vector to a string, using `,` as separator.

        :param np.ndarray x: the vector
        :return: the string
        :rtype: str
        """
        return ",".join([float_to_str(xx) for xx in x])

    def is_equal(self, x1: np.ndarray, x2: np.ndarray) -> bool:
        """
        Check if two vectors are equal.

        :param np.ndarray x1: the first vector
        :param np.ndarray x2: the second
        :return: `True` if the contents of both vectors are equal,
            `False` otherwise
        :rtype: bool
        """
        return np.array_equal(x1, x2)

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
        if not (isinstance(x, np.ndarray)):
            raise TypeError(
                f"x must be an numpy.ndarray, but is a {type(x)}.")
        if x.dtype != self.dtype:
            raise TypeError(
                f"x must be of type {self.dtype} but is of type {x.dtype}.")
        if (len(x.shape) != 1) or (x.shape[0] != self.dimension):
            raise ValueError(f"x must be of shape ({self.dimension}) but is "
                             f"of shape {x.shape}.")
        if not is_all_finite(x):
            raise ValueError("All elements must be finite.")

    def scale(self) -> int:
        """
        Get the number of different floating point values in this space.

        :return: The space contains unrestricted floating point numbers, so we
            return the approximate number of finite floating point numbers.
        :rtype: int
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

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param KeyValueLogger logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_SPACE_NUM_VARS, self.dimension)
        logger.key_value(KEY_NUMPY_TYPE, self.dtype.char)
