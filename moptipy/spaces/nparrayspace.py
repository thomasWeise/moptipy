"""The base class for spaces based on numpy arrays."""
from typing import Final

import numpy as np

from moptipy.api.logging import KEY_SPACE_NUM_VARS
from moptipy.api.space import Space
from moptipy.utils.logger import CSV_SEPARATOR, KeyValueLogSection
from moptipy.utils.nputils import (
    KEY_NUMPY_TYPE,
    array_to_str,
    numpy_type_to_str,
)
from moptipy.utils.types import check_int_range, type_error


class NPArraySpace(Space):
    """
    A space where each element is a one-dimensional :class:`numpy.ndarray`.

    Such spaces can serve as basis for implementing combinatorial
    optimization and can be extended to host permutations.

    >>> import numpy as npx
    >>> s = NPArraySpace(9, npx.dtype(int))
    >>> print(s.dimension)
    9
    >>> print(s.dtype)
    int64
    >>> print(s.create())
    [0 0 0 0 0 0 0 0 0]
    >>> print(s.to_str(s.create()))
    0;0;0;0;0;0;0;0;0
    >>> print(s.from_str(s.to_str(s.create())))
    [0 0 0 0 0 0 0 0 0]
    """

    def __init__(self, dimension: int, dtype: np.dtype) -> None:
        """
        Create the numpy array-based search space.

        :param dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param dtype: the data type
        """
        if not isinstance(dtype, np.dtype):
            raise type_error(dtype, "dtype", np.dtype)
        if (not isinstance(dtype.char, str)) or (len(dtype.char) != 1):
            raise ValueError(
                f"dtype.char must be str of length 1, but is {dtype.char}")
        #: The basic data type of the vector space elements.
        self.dtype: Final[np.dtype] = dtype
        #: The dimension, i.e., the number of elements of the vectors.
        self.dimension: Final[int] = check_int_range(
            dimension, "dimension", 1, 100_000_000)
        # the function forwards
        self.copy = np.copyto  # type: ignore
        self.is_equal = np.array_equal  # type: ignore
        self.to_str = array_to_str  # type: ignore

    def create(self) -> np.ndarray:
        """
        Create a vector with all zeros.

        :return: the vector
        """
        return np.zeros(shape=self.dimension, dtype=self.dtype)

    def from_str(self, text: str) -> np.ndarray:
        """
        Convert a string to a vector.

        :param text: the text
        :return: the vector
        :raises TypeError: if `text` is not a `str`
        :raises ValueError: if `text` cannot be converted to a valid vector
        """
        if not (isinstance(text, str)):
            raise type_error(text, "text", str)
        x = np.fromstring(text, dtype=self.dtype, sep=CSV_SEPARATOR)
        self.validate(x)
        return x

    def validate(self, x: np.ndarray) -> None:
        """
        Validate a numpy nd-array.

        :param x: the numpy vector
        :raises TypeError: if the string is not an :class:`numpy.ndarray`.
        :raises ValueError: if the shape or data type of the vector is wrong
            or any of its element is not finite.
        """
        if not isinstance(x, np.ndarray):
            raise type_error(x, "x", np.ndarray)
        if x.dtype != self.dtype:
            raise ValueError(
                f"x must be of type {self.dtype} but is of type {x.dtype}.")
        if (len(x.shape) != 1) or (x.shape[0] != self.dimension):
            raise ValueError(f"x must be of shape ({self.dimension}) but is "
                             f"of shape {x.shape}.")

    def __str__(self) -> str:
        """
        Get the name of this np array space.

        :return: "ndarray" + dimension + dtype.char

        >>> import numpy as npx
        >>> print(NPArraySpace(4, npx.dtype(int)))
        ndarray4l
        """
        return f"ndarray{self.dimension}{self.dtype.char}"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> import numpy as npx
        >>> dt = npx.dtype(float)
        >>> dt.char
        'd'
        >>> space = NPArraySpace(10, dt)
        >>> space.dimension
        10
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         space.log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[-2]
        'dtype: d'
        >>> text[-3]
        'nvars: 10'
        >>> text[-4]
        'class: moptipy.spaces.nparrayspace.NPArraySpace'
        >>> text[-5]
        'name: ndarray10d'
        >>> len(text)
        6
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_SPACE_NUM_VARS, self.dimension)
        logger.key_value(KEY_NUMPY_TYPE, numpy_type_to_str(self.dtype))
