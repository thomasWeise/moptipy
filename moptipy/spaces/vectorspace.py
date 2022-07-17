"""An implementation of an unconstrained n-dimensional continuous space."""

from typing import Final

import numpy

from moptipy.spaces.nparrayspace import NPArraySpace
from moptipy.utils.nputils import is_np_float, is_all_finite

#: The default numerical datatype
_DEFAULT_TYPE: Final[numpy.dtype] = numpy.dtype(numpy.float64)


class VectorSpace(NPArraySpace):
    """
    A vector space where each element is a one-dimensional numpy float array.

    Such spaces are useful for continuous optimization.
    """

    def __init__(self, dimension: int,
                 dtype: numpy.dtype = _DEFAULT_TYPE) -> None:
        """
        Create the vector-based search space.

        :param dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param dtype: The basic data type of the vector space,
            i.e., the type of the decision variables
        """
        super().__init__(dimension, dtype)
        if not is_np_float(self.dtype):
            raise TypeError(f"Invalid data type {dtype}.")

    def validate(self, x: numpy.ndarray) -> None:
        """
        Validate a vector.

        :param x: the real vector
        :raises TypeError: if the vector is not an element of this space.
        :raises ValueError: if the shape of the vector is wrong or any of its
            element is not finite.
        """
        super().validate(x)
        if not is_all_finite(x):
            raise ValueError("All elements must be finite.")

    def n_points(self) -> int:
        """
        Get the number of different floating point values in this space.

        :return: The space contains unrestricted floating point numbers, so we
            return the approximate number of finite floating point numbers.

        >>> print(VectorSpace(3, numpy.dtype(numpy.float64)).n_points())
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

    def __str__(self) -> str:
        """
        Get the name of this space.

        :return: "vector" + dimension + dtype.char

        >>> print(VectorSpace(3, numpy.dtype(numpy.float64)))
        vector3d
        """
        return f"vector{self.dimension}{self.dtype.char}"
