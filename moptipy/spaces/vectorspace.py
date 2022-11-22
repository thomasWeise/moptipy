"""An implementation of an box-constrained n-dimensional continuous space."""

from math import isfinite
from typing import Callable, Final, Iterable, cast

import numpy
from numpy import clip

from moptipy.spaces.nparrayspace import NPArraySpace
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_FLOAT, array_to_str, is_np_float
from moptipy.utils.types import type_error

#: the log key for the lower bound, i.e., the minimum permitted value
KEY_LOWER_BOUND: Final[str] = "lb"
#: the log key for the upper bound, i.e., the maximum permitted value
KEY_UPPER_BOUND: Final[str] = "ub"


class VectorSpace(NPArraySpace):
    """
    A vector space where each element is a n-dimensional real vector.

    Such spaces are useful for continuous optimization. The vectors are
    implemented as one-dimensional `numpy.ndarray`s of length `n`.
    A vector space is constraint by a box which defines the minimum and
    maximum permitted value for each of its `n` elements.
    """

    def __init__(self, dimension: int,
                 lower_bound: float | Iterable[float] = 0.0,
                 upper_bound: float | Iterable[float] = 1.0,
                 dtype: numpy.dtype = DEFAULT_FLOAT) -> None:
        """
        Create the vector-based search space.

        :param dimension: The dimension of the search space,
            i.e., the number of decision variables.
        :param dtype: The basic data type of the vector space,
            i.e., the type of the decision variables
        :param lower_bound: the optional minimum value(s)
        :param upper_bound: the optional maximum value(s)
        """
        super().__init__(dimension, dtype)
        if not is_np_float(dtype):
            raise TypeError(f"Invalid data type {dtype}.")

        # first, we process the lower bounds
        lower_bound_all_same: bool = True
        if isinstance(lower_bound, float):
            # only a single value is given
            if not isfinite(lower_bound):
                raise ValueError(
                    f"invalid lower bound {lower_bound}.")
            # if a single value is given, then we expand it a vector
            lower_bound = numpy.full(dimension, lower_bound, dtype)
        elif isinstance(lower_bound, Iterable):
            # lower bounds are given as vector or iterable
            lb = numpy.array(lower_bound, dtype)
            if len(lb) != dimension:
                raise ValueError(f"wrong length {lb} of lower "
                                 f"bound iterable {lower_bound}")
            if lb.shape != (dimension, ):
                raise ValueError(f"invalid shape={lb.shape} of "
                                 f"lower bound {lower_bound}")
            first = lb[0]
            for index, item in enumerate(lb):
                if first != item:
                    lower_bound_all_same = False
                if not numpy.isfinite(item):
                    raise ValueError(f"{index}th lower bound={item}")
            lower_bound = lb
        else:
            raise type_error(lower_bound, "lower_bound", (
                float, Iterable, None))

        # now, we process the upper bounds
        upper_bound_all_same: bool = True
        if isinstance(upper_bound, float):
            # only a single value is given
            if not isfinite(upper_bound):
                raise ValueError(
                    f"invalid upper bound {upper_bound}.")
            # if a single value is given, then we expand it a vector
            upper_bound = numpy.full(dimension, upper_bound, dtype)
        elif isinstance(upper_bound, Iterable):
            # upper bounds are given as vector or iterable
            lb = numpy.array(upper_bound, dtype)
            if len(lb) != dimension:
                raise ValueError(f"wrong length {lb} of upper "
                                 f"bound iterable {upper_bound}")
            if lb.shape != (dimension,):
                raise ValueError(f"invalid shape={lb.shape} of "
                                 f"upper bound {upper_bound}")
            first = lb[0]
            for index, item in enumerate(lb):
                if first != item:
                    upper_bound_all_same = False
                if not numpy.isfinite(item):
                    raise ValueError(f"{index}th upper bound={item}")
            upper_bound = lb
        else:
            raise type_error(upper_bound, "upper_bound", (
                float, Iterable, None))

        # check that the bounds are consistent
        for idx, ll in enumerate(lower_bound):
            if not ll < upper_bound[idx]:
                raise ValueError(f"lower_bound[{idx}]={ll} >= "
                                 f"upper_bound[{idx}]={upper_bound[idx]}")

        #: the lower bounds for all variables
        self.lower_bound: Final[numpy.ndarray] = lower_bound
        #: all dimensions have the same lower bound
        self.lower_bound_all_same: Final[bool] = lower_bound_all_same
        #: the upper bounds for all variables
        self.upper_bound: Final[numpy.ndarray] = upper_bound
        #: all dimensions have the same upper bound
        self.upper_bound_all_same: Final[bool] = upper_bound_all_same

    def clipped(self, func: Callable[[numpy.ndarray], int | float]) \
            -> Callable[[numpy.ndarray], int | float]:
        """
        Wrap a function ensuring that all vectors are clipped to the bounds.

        This function is useful to ensure that only valid vectors are passed
        to :meth:`~moptipy.api.process.Process.evaluate`.

        :param func: the function to wrap
        :returns: the wrapped function
        """
        return cast(Callable[[numpy.ndarray], int | float],
                    lambda x, lb=self.lower_bound, ub=self.upper_bound,
                    ff=func: ff(clip(x, lb, ub, x)))

    def validate(self, x: numpy.ndarray) -> None:
        """
        Validate a vector.

        :param x: the real vector
        :raises TypeError: if the vector is not an element of this space.
        :raises ValueError: if the shape of the vector is wrong or any of its
            element is not finite.
        """
        super().validate(x)

        mib: Final[numpy.ndarray] = self.lower_bound
        mab: Final[numpy.ndarray] = self.upper_bound
        for index, item in enumerate(x):
            miv = mib[index]
            mav = mab[index]
            if not numpy.isfinite(item):
                raise ValueError(f"x[{index}]={item}, which is not finite")
            if not (miv <= item <= mav):
                raise ValueError(
                    f"x[{index}]={item}, but should be in [{miv},{mav}].")

    def n_points(self) -> int:
        """
        Get an upper bound for the number of different values in this space.

        :return: We return the approximate number of finite floating point
          numbers while ignoring the box constraint. This value here therefore
          is an upper bound.

        >>> import numpy as npx
        >>> print(VectorSpace(3, dtype=npx.dtype(npx.float64)).n_points())
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

        :return: "r" + dimension + dtype.char

        >>> import numpy as npx
        >>> print(VectorSpace(3, dtype=npx.dtype(npx.float64)))
        r3d
        """
        return f"r{self.dimension}{self.dtype.char}"

    def log_bounds(self, logger: KeyValueLogSection) -> None:
        """
        Log the bounds of this space to the given logger.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> import numpy as npx
        >>> space = VectorSpace(2, -5.0, [2.0, 3.0])
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         space.log_bounds(kv)
        ...     text = l.get_log()
        >>> text[-2]
        'ub: 2;3'
        >>> text[-3]
        'lb: -5'
        >>> len(text)
        4
        """
        if self.lower_bound_all_same:
            logger.key_value(
                KEY_LOWER_BOUND, self.lower_bound[0], also_hex=False)
        else:
            logger.key_value(KEY_LOWER_BOUND, array_to_str(self.lower_bound))
        if self.upper_bound_all_same:
            logger.key_value(
                KEY_UPPER_BOUND, self.upper_bound[0], also_hex=False)
        else:
            logger.key_value(KEY_UPPER_BOUND, array_to_str(self.upper_bound))

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> import numpy as npx
        >>> space = VectorSpace(2, -5.0, [2.0, 3.0])
        >>> space.dimension
        2
        >>> space.dtype.char
        'd'
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         space.log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[-2]
        'ub: 2;3'
        >>> text[-3]
        'lb: -5'
        >>> text[-4]
        'dtype: d'
        >>> text[-5]
        'nvars: 2'
        >>> text[-6]
        'class: moptipy.spaces.vectorspace.VectorSpace'
        >>> text[-7]
        'name: r2d'
        >>> len(text)
        8
        """
        super().log_parameters_to(logger)
        self.log_bounds(logger)
