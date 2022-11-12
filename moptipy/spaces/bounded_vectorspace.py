"""An implementation of an unconstrained n-dimensional continuous space."""

from math import isfinite
from typing import Final

import numpy

from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import num_to_str_for_name
from moptipy.utils.types import type_error


class BoundedVectorSpace(VectorSpace):
    """
    A bounded vector space whose elements are a one-dimensional numpy array.

    Such spaces are useful for continuous optimization.
    """

    def __init__(self, dimension: int,
                 x_min: float = -1, x_max: float = 1,
                 dtype: numpy.dtype = numpy.dtype(numpy.float64)) -> None:
        """
        Create the vector-based search space.

        :param dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param x_min: the minimum permitted value
        :param x_max: the maximum permitted value
        :param dtype: The basic data type of the vector space,
            i.e., the type of the decision variables
        """
        super().__init__(dimension, dtype)
        if not isinstance(x_min, float):
            raise type_error(x_min, "x_min", float)
        if not isfinite(x_min):
            raise ValueError(f"x_min must be finite, but is {x_min}.")
        if not isinstance(x_max, float):
            raise type_error(x_max, "x_max", float)
        if not isfinite(x_max):
            raise ValueError(f"x_max must be finite, but is {x_max}.")
        if x_min >= x_max:
            raise ValueError(f"x_max > x_min must hold, but got "
                             f"x_min={x_min} and x_max={x_max}.")
        #: the minimum value for each element of the vectors in the space
        self.x_min: Final[float] = x_min
        #: the maximum value for each element of the vectors in the space
        self.x_max: Final[float] = x_max

    def validate(self, x: numpy.ndarray) -> None:
        """
        Validate a vector.

        :param x: the real vector
        :raises TypeError: if the vector is not an element of this space.
        :raises ValueError: if the shape of the vector is wrong or any of its
            element is not finite.
        """
        super().validate(x)
        xmin: Final[float] = x.min()
        xmax: Final[float] = x.max()
        if not (self.x_min <= xmin <= xmax <= self.x_max):
            raise ValueError(f"permitted range of elements is [{self.x_min}"
                             f", {self.x_max}] but found elements in range "
                             f"[{xmin}, {xmax}].")

    def __str__(self) -> str:
        """
        Get the name of this space.

        :return: "vector" + dimension + dtype.char

        >>> print(BoundedVectorSpace(3, -1.0, 1.0))
        vector3d_m1_1

        >>> print(BoundedVectorSpace(3, -1.0, 1.6))
        vector3d_m1_1d6
        """
        return f"{super().__str__()}_{num_to_str_for_name(self.x_min)}_" \
               f"{num_to_str_for_name(self.x_max)}"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("xMin", self.x_min, also_hex=True)
        logger.key_value("xMax", self.x_max, also_hex=True)
