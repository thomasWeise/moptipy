"""An implementation of an unconstrained n-dimensional continuous space."""

from typing import Final

import numpy

from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.bounds import FloatBounds
from moptipy.utils.logger import KeyValueLogSection


class BoundedVectorSpace(VectorSpace, FloatBounds):
    """
    A bounded vector space whose elements are a one-dimensional numpy array.

    Such spaces are useful for continuous optimization.
    """

    def __init__(self, dimension: int,
                 min_value: float = -1.0, max_value: float = 1.0,
                 dtype: numpy.dtype = numpy.dtype(numpy.float64)) -> None:
        """
        Create the vector-based search space.

        :param dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param min_value: the minimum permitted value
        :param max_value: the maximum permitted value
        :param dtype: The basic data type of the vector space,
            i.e., the type of the decision variables
        """
        VectorSpace.__init__(self, dimension, dtype)
        FloatBounds.__init__(self, min_value, max_value)

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
        if not (self.min_value <= xmin <= xmax <= self.max_value):
            raise ValueError(
                f"permitted range of elements is [{self.min_value}"
                f", {self.max_value}] but found elements in range "
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
        return f"{VectorSpace.__str__(self)}_{FloatBounds.__str__(self)}"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param logger: the logger for the parameters
        """
        VectorSpace.log_parameters_to(self, logger)
        FloatBounds.log_parameters_to(self, logger)  # type: ignore
