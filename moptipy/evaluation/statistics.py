"""A simple and immutable basic statistics record."""

import statistics
from dataclasses import dataclass
from math import gcd, inf, isfinite, sqrt
from typing import Callable, Final, Iterable, cast

from moptipy.utils.logger import CSV_SEPARATOR, SCOPE_SEPARATOR
from moptipy.utils.math import (
    DBL_INT_LIMIT_P,
    try_int,
    try_int_div,
    try_int_root,
)
from moptipy.utils.strings import (
    num_to_str,
    str_to_intfloat,
    str_to_intfloatnone,
)
from moptipy.utils.types import type_error

#: The limit until which we simplify geometric mean data.
_INT_ROOT_LIMIT: Final[int] = int(sqrt(DBL_INT_LIMIT_P))
_ULP: Final[float] = 1 - (2 ** (-53))

#: The minimum value key.
KEY_MINIMUM: Final[str] = "min"
#: The median value key.
KEY_MEDIAN: Final[str] = "med"
#: The arithmetic mean value key.
KEY_MEAN_ARITH: Final[str] = "mean"
#: The geometric mean value key.
KEY_MEAN_GEOM: Final[str] = "geom"
#: The maximum value key.
KEY_MAXIMUM: Final[str] = "max"
#: The standard deviation value key.
KEY_STDDEV: Final[str] = "sd"

#: The number of CSV columns.
CSV_COLS: Final[int] = 6

#: The empty csv row of statistics
EMPTY_CSV_ROW: Final[str] = CSV_SEPARATOR * (CSV_COLS - 1)

#: the internal getters
_GETTERS: Final[dict[str, Callable[["Statistics"],
                                   int | float | None]]] = {
    KEY_MINIMUM: lambda s: s.minimum,
    KEY_MEDIAN: lambda s: s.median,
    KEY_MEAN_ARITH: lambda s: s.mean_arith,
    KEY_MEAN_GEOM: lambda s: s.mean_geom,
    KEY_MAXIMUM: lambda s: s.maximum,
    KEY_STDDEV: lambda s: s.stddev
}


@dataclass(frozen=True, init=False, order=True)
class Statistics:
    """An immutable record with statistics of one quantity."""

    #: The minimum.
    minimum: int | float
    #: The median.
    median: int | float
    #: The arithmetic mean value.
    mean_arith: int | float
    #: The geometric mean value, if defined.
    mean_geom: int | float | None
    #: The maximum.
    maximum: int | float
    #: The standard deviation.
    stddev: int | float

    def __init__(self, n: int,
                 minimum: int | float,
                 median: int | float,
                 mean_arith: int | float,
                 mean_geom: int | float | None,
                 maximum: int | float,
                 stddev: int | float):
        """
        Initialize the statistics class.

        :param minimum: the minimum
        :param median: the median
        :param mean_arith: the arithmetic mean
        :param mean_geom: the geometric mean, or `None` if it is undefined
        :param maximum: the maximum
        :param stddev: the standard deviation (`0` is also used for undefined)
        """
        if not isinstance(n, int):
            raise type_error(n, "n", int)
        if n <= 0:
            raise ValueError(f"n must be >= 1, but is {n}.")

        # check minimum
        if not isinstance(minimum, (int, float)):
            raise type_error(minimum, "minimum", (int, float))
        if isinstance(minimum, float) and (not isfinite(minimum)):
            raise ValueError(f"minimum must be finite, but is {minimum}.")

        # check median
        if not isinstance(median, (int, float)):
            raise type_error(median, "median", (int, float))
        if isinstance(median, float) and (not isfinite(median)):
            raise ValueError(f"med must be finite, but is {median}.")
        if n == 1:
            if median != minimum:
                raise ValueError(f"median ({median}) must equal "
                                 f"minimum ({minimum}) if n=1.")
        elif median < minimum:
            raise ValueError(
                f"median ({median}) must be >= minimum ({minimum}) if n>1.")

        # check maximum
        if not isinstance(maximum, (int, float)):
            raise type_error(maximum, "maximum", (int, float))
        if isinstance(maximum, float) and (not isfinite(maximum)):
            raise ValueError(f"maximum must be finite, but is {maximum}.")
        if n == 1:
            if maximum != minimum:
                raise ValueError(f"maximum ({maximum}) must equal "
                                 f"minimum ({minimum}) if n=1.")
        elif maximum < median:
            raise ValueError(
                f"maximum ({maximum}) must be >= med ({median}) if n>1.")

        # check arithmetic mean
        if not isinstance(mean_arith, (int, float)):
            raise type_error(mean_arith, "mean_arith", (int, float))
        if isinstance(mean_arith, float) and (not isfinite(mean_arith)):
            raise ValueError(
                f"mean_arith must be finite, but is {mean_arith}.")
        if n == 1:
            if mean_arith != minimum:
                raise ValueError(f"mean_arith ({mean_arith}) must equal "
                                 f"minimum ({minimum}) if n=1.")
        else:
            if mean_arith < minimum:
                raise ValueError(
                    f"mean_arith ({mean_arith}) must be >= "
                    f"minimum ({minimum}) if n>1.")
            if mean_arith > maximum:
                raise ValueError(
                    f"mean_arith ({mean_arith}) must be <= "
                    f"maximum ({maximum}) if n>1.")

        # check geometric mean
        if mean_geom is None:
            if minimum > 0:
                raise ValueError(
                    f"If minimum ({minimum}) > 0, then mean_geom must be "
                    f"defined, but it is {mean_geom}.")
        else:
            if minimum <= 0:
                raise ValueError(
                    f"If minimum ({minimum}) <= 0, then mean_geom is "
                    f"undefined, but it is {mean_geom}.")
            if not isinstance(mean_geom, (int, float)):
                raise type_error(mean_geom, "mean_geom", (int, float))
            if isinstance(mean_geom, float) and (not isfinite(mean_geom)):
                raise ValueError(
                    f"mean_geom must be finite, but is {mean_geom}.")
            if n == 1:
                if mean_geom != minimum:
                    raise ValueError(f"mean_geom ({mean_geom}) must equal "
                                     f"minimum ({minimum}) if n=1.")
            else:
                if mean_geom < minimum:
                    raise ValueError(
                        f"mean_geom ({mean_geom}) must be >= "
                        f"minimum ({minimum}) if n>1.")
                if mean_geom > mean_arith:
                    raise ValueError(
                        f"mean_geom ({mean_geom}) must be <= "
                        f"mean_arith ({mean_arith}) if n>1.")
                if (mean_geom >= mean_arith) and (minimum < maximum):
                    raise ValueError(
                        f"mean_geom ({mean_geom}) must be < "
                        f"mean_arith ({mean_arith}) if n>1 and "
                        f"minimum ({minimum}) < maximum ({maximum}).")

        if not isinstance(stddev, (int, float)):
            raise type_error(stddev, "stddev", (int, float))
        if isinstance(stddev, float) and (not isfinite(stddev)):
            raise ValueError(f"stddev must be finite, but is {stddev}.")

        if n == 1:
            if stddev != 0:
                raise ValueError(
                    f"If n==1, stddev must be 0, but is {stddev}.")
            stddev = 0  # set it to int 0

        if (stddev == 0) ^ (minimum == maximum):
            raise ValueError(
                f"If stddev ({stddev}) is 0, then minimum ({minimum}) must "
                f"equal maximum ({maximum}) and vice versa.")
        if minimum == maximum:
            stddev = 0  # set if to int 0

        # fix types to int where possible without loss of precision

        object.__setattr__(self, "minimum", try_int(minimum))
        object.__setattr__(self, "median", try_int(median))
        object.__setattr__(self, "maximum", try_int(maximum))
        object.__setattr__(self, "mean_arith", try_int(mean_arith))
        object.__setattr__(self, "mean_geom",
                           None if mean_geom is None else try_int(mean_geom))
        object.__setattr__(self, "stddev", try_int(stddev))

    def min_mean(self) -> int | float:
        """
        Obtain the smallest of the three mean values.

        :return: the smallest of `mean_arith`, `mean_geom`, and `median`
        :rtype: Union[int, float]
        """
        if self.mean_geom is None:
            if self.mean_arith < self.median:
                return self.mean_arith
            return self.median

        if self.mean_geom < self.median:
            return self.mean_geom
        return self.median

    def max_mean(self) -> int | float:
        """
        Obtain the largest of the three mean values.

        :return: the largest of `mean_arith`, `mean_geom`, and `median`
        :rtype: Union[int, float]
        """
        if self.mean_arith > self.median:
            return self.mean_arith
        return self.median

    @staticmethod
    def create(source: Iterable[int | float]) -> "Statistics":
        """
        Create a statistics object from an iterable.

        :param source: the source
        :return: a statistics representing the statistics over `source`
        :rtype: Statistics

        >>> from moptipy.evaluation.statistics import Statistics
        >>> s = Statistics.create([3, 1, 2, 5])
        >>> print(s.minimum)
        1
        >>> print(s.maximum)
        5
        >>> print(s.mean_arith)
        2.75
        >>> print(s.median)
        2.5
        >>> print(f"{s.mean_geom:.4f}")
        2.3403
        >>> print(f"{s.min_mean():.4f}")
        2.3403
        >>> print(f"{s.max_mean()}")
        2.75
        """
        if not isinstance(source, Iterable):
            raise type_error(source, "source", Iterable)

        minimum: int | float = inf
        maximum: int | float = -inf
        can_int: bool = True  # are all values integers?
        int_sum: int = 0  # the integer sum (for mean, stddev)
        int_prod: int = 1  # the integer product (for geom_mean)
        int_sum_sqr: int = 0  # the sum of squares (for stddev)
        n: int = 0  # the number of items

        for e in source:  # iterate over all data
            n = n + 1
            e = try_int(e)

            if can_int:  # can we do integers
                if not isinstance(e, int):
                    can_int = False
            if can_int:  # so far, we only encountered ints
                zz = cast(int, e)
                int_sum += zz  # so we can sum exactly
                int_prod *= zz  # and can compute exact products
                int_sum_sqr += zz * zz

            if e < minimum:
                minimum = e  # update minimum
            if e > maximum:
                maximum = e  # update maximum

        if n <= 0:
            raise ValueError("source cannot be empty.")

        if minimum >= maximum:
            return Statistics(n=n,
                              minimum=minimum,
                              median=minimum,
                              mean_arith=minimum,
                              mean_geom=None if minimum <= 0 else minimum,
                              maximum=maximum,
                              stddev=0)

        stddev: int | float = 0
        mean_geom: int | float | None = None
        mean_arith: int | float

        if can_int:  # if we get here, we have exact sums and product
            mean_arith = try_int_div(int_sum, n)

            if n > 1:  # standard deviation only defined for n > 1
                int_sum2: int = (int_sum * int_sum)
                i_gcd = gcd(int_sum2, n)
                int_sum2 = int_sum2 // i_gcd
                i_n = n // i_gcd

                var: int | float  # the container for the variance
                if i_n == 1:
                    var = try_int_div(int_sum_sqr - int_sum2, n - 1)
                else:  # variance is float
                    var = (int_sum_sqr - (int_sum2 / i_n)) / (n - 1)

                stddev = try_int(sqrt(var))

            if minimum > 0:  # geometric mean only defined for all-positive
                if int_prod == 0:
                    mean_geom = 0  # geometric mean is 0 if product is 0
                else:  # if not, geom_mean = prod ** (1/n)
                    mean_geom = try_int_root(int_prod, n, True)
                    if mean_geom is None:
                        mean_geom = statistics.geometric_mean(source)
        else:  # ok, we do not have only integer-like values
            mean_arith = try_int(statistics.mean(source))
            if minimum > 0:
                mean_geom = try_int(statistics.geometric_mean(source))
            if n > 1:
                stddev = try_int(statistics.stdev(source))

        if mean_geom is not None:
            # Deal with errors that may have arisen due to
            # numerical imprecision.
            if mean_geom < minimum:
                if (mean_geom / _ULP) >= (minimum * _ULP):
                    mean_geom = minimum
            if mean_geom > maximum:
                if (maximum / _ULP) >= (mean_geom * _ULP):
                    mean_geom = maximum

        return Statistics(n=n,
                          minimum=minimum,
                          median=statistics.median(source),
                          mean_arith=mean_arith,
                          mean_geom=mean_geom,
                          maximum=maximum,
                          stddev=stddev)

    @staticmethod
    def csv_col_names(prefix: str) -> Iterable[str]:
        """
        Make the column names suitable for a CSV-formatted file.

        :param str prefix: the prefix name of the columns
        :return: the column header strings
        :rtype: Iterable[str]
        """
        return [f"{prefix}{SCOPE_SEPARATOR}{b}"
                for b in [KEY_MINIMUM, KEY_MEDIAN,
                          KEY_MEAN_ARITH, KEY_MEAN_GEOM,
                          KEY_MAXIMUM, KEY_STDDEV]]

    @staticmethod
    def value_to_csv(value: int | float) -> str:
        """
        Expand a single value to a CSV row.

        :param Union[int, float] value: the value
        :return: the CSV row.
        :rtype: str
        """
        s: str = f"{num_to_str(value)}{CSV_SEPARATOR}"
        return f"{s * 5}0"

    def to_csv(self) -> str:
        """
        Generate a string with the data of this record in CSV format.

        :return: the string
        :rtype: str
        """
        q: Final[str] = "" if (self.mean_geom is None) \
            else num_to_str(self.mean_geom)
        return f"{num_to_str(self.minimum)}{CSV_SEPARATOR}" \
               f"{num_to_str(self.median)}{CSV_SEPARATOR}" \
               f"{num_to_str(self.mean_arith)}" \
               f"{CSV_SEPARATOR}" \
               f"{q}" \
               f"{CSV_SEPARATOR}" \
               f"{num_to_str(self.maximum)}{CSV_SEPARATOR}" \
               f"{num_to_str(self.stddev)}"

    @staticmethod
    def from_csv(n: int, row: str | Iterable[str]) -> "Statistics":
        """
        Convert a CSV string (or separate CSV cells) to a Statistics object.

        :param int n: the number of observations
        :param Union[str, Iterable[str]] row: either the single string or
            the iterable of separate strings
        :return: the :class:`Statistics` instance
        :rtype: Statistics
        """
        cells = row.split(CSV_SEPARATOR) \
            if isinstance(row, str) else row

        mini, med, mean, geo, maxi, sd = cells
        return Statistics(n,
                          str_to_intfloat(mini),
                          str_to_intfloat(med),
                          str_to_intfloat(mean),
                          str_to_intfloatnone(geo),
                          str_to_intfloat(maxi),
                          str_to_intfloat(sd))

    @staticmethod
    def getter(dimension: str) -> Callable[["Statistics"],
                                           int | float | None]:
        """
        Produce a function that obtains the given dimension from Statistics.

        :param dimension: the dimension
        :returns: a callable that returns the value corresponding to the
            dimension
        """
        if not isinstance(dimension, str):
            raise type_error(dimension, "dimension", str)
        if dimension in _GETTERS:
            return _GETTERS[dimension]
        raise ValueError(f"unknown dimension '{dimension}', "
                         f"should be one of {sorted(_GETTERS.keys())}.")
