"""A simple and immutable basic statistics record."""

import statistics
from dataclasses import dataclass
from math import isfinite, sqrt, gcd, inf
from typing import Union, Iterable, Final, cast

from moptipy.evaluation._utils import _DBL_INT_LIMIT_P, _try_int, _try_div
from moptipy.evaluation._utils import _str_to_if, _str_to_ifn
from moptipy.utils import logging

#: The limit until which we simplify geometric mean data.
_INT_ROOT_LIMIT: Final[int] = int(sqrt(_DBL_INT_LIMIT_P))
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
EMPTY_CSV_ROW: Final[str] = logging.CSV_SEPARATOR * (CSV_COLS - 1)


@dataclass(frozen=True, init=False, order=True)
class Statistics:
    """An immutable record with statistics of one quantity."""

    #: The minimum.
    minimum: Union[int, float]
    #: The median.
    median: Union[int, float]
    #: The arithmetic mean value.
    mean_arith: Union[int, float]
    #: The geometric mean value, if defined.
    mean_geom: Union[int, float, None]
    #: The maximum.
    maximum: Union[int, float]
    #: The standard deviation.
    stddev: Union[int, float]

    def __init__(self, n: int,
                 minimum: Union[int, float],
                 median: Union[int, float],
                 mean_arith: Union[int, float],
                 mean_geom: Union[int, float, None],
                 maximum: Union[int, float],
                 stddev: Union[int, float]):
        """
        Initialize the statistics class.

        :param Union[int, float] minimum: the minimum
        :param Union[int, float] median: the median
        :param Union[int, float] mean_arith: the arithmetic mean
        :param Union[int, float, None] mean_geom: the geometric mean
        :param Union[int, float] maximum: the maximum
        :param Union[int, float] stddev: the standard deviation
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be int but is {type(n)}.")
        if n <= 0:
            raise ValueError(f"n must be >= 1, but is {n}.")

        # check minimum
        if not isinstance(minimum, (int, float)):
            raise TypeError(
                f"minimum must be float or int, but is {type(minimum)}.")
        if isinstance(minimum, float) and (not isfinite(minimum)):
            raise ValueError(f"minimum must be finite, but is {minimum}.")

        # check median
        if not isinstance(median, (int, float)):
            raise TypeError(
                f"median must be int or float, but is {type(median)}.")
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
            raise TypeError(
                f"maximum must be float or int, but is {type(maximum)}.")
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
            raise TypeError("mean_arith must be int or float, "
                            f"but is {type(mean_arith)}.")
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
                raise TypeError("mean_geom must be int or float, "
                                f"but is {type(mean_geom)}.")
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
                if mean_geom > maximum:
                    raise ValueError(
                        f"mean_geom ({mean_geom}) must be <= "
                        f"maximum ({maximum}) if n>1.")

        if not isinstance(stddev, (int, float)):
            raise TypeError(
                f"stddev must be int or float, but is {type(stddev)}.")
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

        object.__setattr__(self, "minimum", _try_int(minimum))
        object.__setattr__(self, "median", _try_int(median))
        object.__setattr__(self, "maximum", _try_int(maximum))
        object.__setattr__(self, "mean_arith", _try_int(mean_arith))
        object.__setattr__(self, "mean_geom",
                           None if mean_geom is None else _try_int(mean_geom))
        object.__setattr__(self, "stddev", _try_int(stddev))

    def min_mean(self) -> Union[int, float]:
        """
        Obtain the smallest of the three mean values.

        :return: the smallest of `mean_arith`, `mean_geom`, and `median`
        :rtype: Union[int, float]
        """
        if (self.mean_geom is None) or \
                (self.mean_arith <= self.mean_geom):
            if self.mean_arith <= self.median:
                return self.mean_arith
            return self.median

        if self.mean_geom <= self.median:
            return self.mean_geom
        return self.median

    def max_mean(self) -> Union[int, float]:
        """
        Obtain the largest of the three mean values.

        :return: the largest of `mean_arith`, `mean_geom`, and `median`
        :rtype: Union[int, float]
        """
        if (self.mean_geom is None) or \
                (self.mean_arith >= self.mean_geom):
            if self.mean_arith >= self.median:
                return self.mean_arith
            return self.median
        if self.mean_geom >= self.median:
            return self.mean_geom
        return self.median

    @staticmethod
    def create(source: Iterable[Union[int, float]]) -> 'Statistics':
        """
        Create a statistics object from an iterable.

        :param Collection[Union[int,float]] source: the source
        :return: a statistics representing the statistics over `source`
        :rtype: Statistics
        """
        if not isinstance(source, Iterable):
            raise TypeError(
                f"source must be Iterable, but is {type(source)}.")

        minimum: Union[int, float] = inf
        maximum: Union[int, float] = -inf
        can_int: bool = True  # are all values integers?
        int_sum: int = 0  # the integer sum (for mean, stddev)
        int_prod: int = 1  # the integer product (for geom_mean)
        int_sum_sqr: int = 0  # the sum of squares (for stddev)
        n: int = 0  # the number of items

        for e in source:  # iterate over all data
            n = n + 1
            e = _try_int(e)

            if can_int:  # can we do integers
                if not isinstance(e, int):
                    can_int = False
            if can_int:  # so far, we only encountered ints
                int_sum += cast(int, e)  # so we can sum exactly
                int_prod *= cast(int, e)  # and can compute exact products
                int_sum_sqr += cast(int, e * e)

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

        stddev: Union[int, float] = 0
        mean_geom: Union[int, float, None] = None
        mean_arith: Union[int, float]

        if can_int:  # if we get here, we have exact sums and product
            mean_arith = _try_div(int_sum, n)

            if n > 1:  # standard deviation only defined for n > 1
                int_sum2: int = (int_sum * int_sum)
                i_gcd = gcd(int_sum2, n)
                int_sum2 = int_sum2 // i_gcd
                i_n = n // i_gcd

                var: Union[int, float]  # the container for the variance
                if i_n == 1:
                    var = _try_div(int_sum_sqr - int_sum2, n - 1)
                else:  # variance is float
                    var = (int_sum_sqr - (int_sum2 / i_n)) / (n - 1)

                stddev = sqrt(var)

            if minimum > 0:  # geometric mean only defined for all-positive
                if int_prod == 0:
                    mean_geom = 0  # geometric mean is 0 if product is 0
                else:  # if not, geom_mean = prod ** (1/n)
                    # We try to remove factors of form (1/n) and store them in
                    # base_mul. We need to prevent overflow when converting to
                    # float. Right now, I only have the idea of brute force
                    # removal of any factor j that is an integer i ** n.
                    # Such a factor will be a power factor j in int_prod and
                    # an integer factor i in base_mul.
                    # After dividing int_prod by as many such factors as
                    # possible, we can compute int_prod ** 1/n and multiply the
                    # result with base_mul.
                    frac = 1.0 / n
                    i = max(1, min(100_000, int(int_prod ** frac)))
                    base_mul = 1
                    while (i > 1) and (int_prod > _INT_ROOT_LIMIT):
                        j = i ** n
                        got: bool = False
                        while (j < int_prod) and ((int_prod % j) == 0):
                            base_mul *= i
                            int_prod //= j
                            got = True
                        if got:
                            i = min(i - 1, int(int_prod ** frac))
                        else:
                            i = i - 1

                    if int_prod == 1:  # if this holds, we do not need to
                        mean_geom = base_mul  # compute the root
                    else:  # otherwise, we may have prevented overflow(?)
                        if int_prod <= _DBL_INT_LIMIT_P:  # no overflow?
                            mean_geom = base_mul * (int_prod ** frac)
                        else:  # too dangerous, there may be overflow
                            mean_geom = statistics.geometric_mean(source)
        else:  # ok, we do not have only integer-like values
            mean_arith = statistics.mean(source)
            if minimum > 0:
                mean_geom = statistics.geometric_mean(source)
            if n > 1:
                stddev = statistics.stdev(source)

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
        return [f"{prefix}{logging.SCOPE_SEPARATOR}{b}"
                for b in [KEY_MINIMUM, KEY_MEDIAN,
                          KEY_MEAN_ARITH, KEY_MEAN_GEOM,
                          KEY_MAXIMUM, KEY_STDDEV]]

    @staticmethod
    def value_to_csv(value: Union[int, float]) -> str:
        """
        Expand a single value to a CSV row.

        :param Union[int, float] value: the value
        :return: the CSV row.
        :rtype: str
        """
        s: str = f"{logging.num_to_str(value)}{logging.CSV_SEPARATOR}"
        return f"{s * 5}0"

    def to_csv(self) -> str:
        """
        Generate a string with the data of this record in CSV format.

        :return: the string
        :rtype: str
        """
        q: Final[str] = "" if (self.mean_geom is None) \
            else logging.num_to_str(self.mean_geom)
        return f"{logging.num_to_str(self.minimum)}{logging.CSV_SEPARATOR}" \
               f"{logging.num_to_str(self.median)}{logging.CSV_SEPARATOR}" \
               f"{logging.num_to_str(self.mean_arith)}" \
               f"{logging.CSV_SEPARATOR}" \
               f"{q}" \
               f"{logging.CSV_SEPARATOR}" \
               f"{logging.num_to_str(self.maximum)}{logging.CSV_SEPARATOR}" \
               f"{logging.num_to_str(self.stddev)}"

    @staticmethod
    def from_csv(n: int, row: Union[str, Iterable[str]]) -> 'Statistics':
        """
        Convert a CSV string (or separate CSV cells) to a Statistics object.

        :param int n: the number of observations
        :param Union[str, Iterable[str]] row: either the single string or
            the iterable of separate strings
        :return: the :class:`Statistics` instance
        :rtype: Statistics
        """
        cells = row.split(logging.CSV_SEPARATOR) \
            if isinstance(row, str) else row

        mini, med, mean, geo, maxi, sd = cells
        return Statistics(n,
                          _str_to_if(mini),
                          _str_to_if(med),
                          _str_to_if(mean),
                          _str_to_ifn(geo),
                          _str_to_if(maxi),
                          _str_to_if(sd))
