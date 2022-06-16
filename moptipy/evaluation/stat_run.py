"""Statistic runs are time-depending statistics over several runs."""
from dataclasses import dataclass
from math import erf, sqrt
from typing import Final, Optional, Iterable, Callable, List, Tuple, Dict, \
    Any, Union

import numba  # type: ignore
import numpy as np

import moptipy.evaluation.statistics as statn
from moptipy.evaluation.base import MultiRun2DData, MultiRunData, \
    PerRunData
from moptipy.evaluation.progress import Progress
from moptipy.utils.nputils import DEFAULT_FLOAT, DEFAULT_INT
from moptipy.utils.nputils import is_np_float
from moptipy.utils.types import type_error

#: The value of the CDF of the standard normal distribution CDF at -1,
#: which corresponds to "mean - 1 * sd".
__Q159: Final[float] = (1.0 + erf(-1.0 / sqrt(2.0))) / 2.0

#: The value of the CDF of the standard normal distribution CDF at +1,
#: which corresponds to "mean + 1 * sd".
__Q841: Final[float] = (1.0 + erf(1.0 / sqrt(2.0))) / 2.0


@numba.njit
def _unique_floats_1d(data: Tuple[np.ndarray, ...]) -> np.ndarray:
    """
    Get all unique values that are >= than the minimum of all arrays.

    :param data: the data
    :return: the `ndarray` with the sorted, unique values
    """
    res: np.ndarray = np.unique(np.concatenate(data).astype(DEFAULT_FLOAT))
    mini = res[0]  # old version: int = -9223372036854775808
    for d in data:
        mini2 = d[0]
        if d[0] > mini:
            mini = mini2
    i: Final[int] = int(np.searchsorted(res, mini))
    if i > 0:
        return res[i:]
    return res


@numba.njit(inline='always')
def __apply_fun(x_unique: np.ndarray,
                x_raw: Tuple[np.ndarray, ...],
                y_raw: Tuple[np.ndarray, ...],
                stat_func: Callable,
                out_len: int,
                dest_y: np.ndarray,
                stat_dim: int,
                values_buf: np.ndarray,
                pos_buf: np.ndarray) -> np.ndarray:
    """
    Perform the work of computing the time-depending statistic.

    The unique x-values `x_unique` have separate been computed with
    :meth:`_unique_floats_1d` from `x_raw` so that they can be reused.
    `x_raw` and `y_raw` are tuples with the raw time and objective data,
    respectively. `stat_fun` is the statistic function that will be applied to
    the step-wise generated data filled into `values_buf`. `pos_buf` will be
    used maintain the current indices into `x_raw` and `y_raw`. `dest_y` will
    be filled with the compute statistic for each element of `x_unique`.
    In a final step, we will remove all redundant elements of both arrays: If
    `x_unique` increases but `dest_y` remains the same, then the corresponding
    point is deleted if it is not the last point in the list. As a result,
    a two-dimensional time/value array is returned.

    :param x_unique: the unique time coordinates
    :param x_raw: a tuple of several x-data arrays
    :param y_raw: a tuple of several y-data arrays
    :param Callable stat_func: a statistic function which must have been
        jitted with numba
    :param out_len: the length of `dest_y` and `x_unique`
    :param dest_y: the destination array for the computed statistics
    :param stat_dim: the dimension of the tuples `x_raw` and `y_raw`
    :param values_buf: the buffer for the values to be passed to `stat_func`
    :param pos_buf: the position buffer
    :return: the two-dimensional `np.ndarray` where the first column is the
        time and the second column is the statistic value
    """
    for i in range(out_len - 1, -1, -1):  # reverse iteration
        x = x_unique[i]  # x_unique holds all unique x values
        for j in range(stat_dim):  # for all Progress datasets do
            idx = pos_buf[j]  # get the current position
            if x < x_raw[j][idx]:  # if x < then current time value
                idx = idx - 1  # step back by one
                pos_buf[j] = idx  # now x >= x_raw[j][idx]
            values_buf[j] = y_raw[j][idx]
        dest_y[i] = stat_func(values_buf)

    changes = 1 + np.flatnonzero(dest_y[1:] != dest_y[:-1])
    ll = len(dest_y) - 1
    if changes[-1] != ll:
        indexes = np.concatenate((np.array([0]), changes, np.array([ll])))
    else:
        indexes = np.concatenate((np.array([0]), changes))
    return np.column_stack((x_unique[indexes], dest_y[indexes]))


@numba.jit(forceobj=True)
def _apply_fun(x_unique: np.ndarray,
               x_raw: Tuple[np.ndarray, ...],
               y_raw: Tuple[np.ndarray, ...],
               stat_func: Callable) -> np.ndarray:
    """
    Compute a time-depending statistic.

    The unique x-values `x_unique` have separate been computed with
    `_unique_floats_1d` from `x_raw` so that they can be reused.
    `x_raw` and `y_raw` are tuples with the raw time and objective data,
    respectively. `stat_fun` is the statistic function that will be applied.
    In a final step, we will remove all redundant elements of both arrays: If
    `x_unique` increases but `dest_y` remains the same, then the corresponding
    point is deleted if it is not the last point in the list. As a result,
    a two-dimensional time/value array is returned. This function uses
    :meth:`__apply_fun` as internal work horse.

    :param x_unique: the unique time coordinates
    :param x_raw: a tuple of several x-data arrays
    :param y_raw: a tuple of several y-data arrays
    :param stat_func: a statistic function which must have been jitted with
        numba
    :return: the two-dimensional `numpy.ndarray` where the first column is the
        time and the second column is the statistic value
    """
    out_len: Final[int] = len(x_unique)
    dest_y: Final[np.ndarray] = np.zeros(out_len, DEFAULT_FLOAT)
    stat_dim: Final[int] = len(x_raw)
    values: Final[np.ndarray] = np.zeros(stat_dim, DEFAULT_FLOAT)
    pos: Final[np.ndarray] = np.array([len(x) - 1 for x in x_raw], DEFAULT_INT)

    return __apply_fun(x_unique, x_raw, y_raw, stat_func, out_len,
                       dest_y, stat_dim, values, pos)


@numba.njit(parallel=True)
def __stat_arith_mean(data: np.ndarray) -> np.number:
    """
    Compute the arithmetic mean.

    :param data: the data
    :return: the arithmetic mean
    """
    return data.mean()


@numba.njit(parallel=True)
def __stat_geo_mean(data: np.ndarray) -> np.number:
    """
    Compute the geometric mean.

    :param data: the data
    :return: the geometric mean
    """
    return np.exp(np.mean(np.log(data)))


@numba.njit(parallel=True)
def __stat_min(data: np.ndarray) -> np.number:
    """
    Compute the minimum.

    :param data: the data
    :return: the minimum
    """
    return data.min()


@numba.njit(parallel=True)
def __stat_max(data: np.ndarray) -> np.number:
    """
    Compute the maximum.

    :param data: the data
    :return: the maximum
    """
    return data.max()


@numba.njit
def __stat_median(data: np.ndarray):
    """
    Compute the median.

    :param data: the data
    :return: the median
    """
    return np.median(data)


@numba.njit(parallel=True)
def __stat_sd(data: np.ndarray) -> np.number:
    """
    Compute the standard deviation.

    :param data: the data
    :return: the standard deviation
    """
    return data.std()


@numba.njit(parallel=True)
def __stat_mean_minus_sd(data: np.ndarray) -> np.number:
    """
    Compute the arithmetic mean minus the standard deviation.

    :param data: the data
    :return: the arithmetic mean minus the standard deviation
    """
    return data.mean() - data.std()


@numba.njit(parallel=True)
def __stat_mean_plus_sd(data: np.ndarray) -> np.number:
    """
    Compute the arithmetic mean plus the standard deviation.

    :param data: the data
    :return: the arithmetic mean plus the standard deviation
    """
    return data.mean() + data.std()


@numba.njit
def __stat_quantile_10(data: np.ndarray):
    """
    Compute the 10% quantile.

    :param data: the data
    :return: the 10% quantile
    """
    length: Final[int] = len(data)
    if (length > 10) and ((length % 10) == 1):
        data.sort()
        return data[(length - 1) // 10]
    return np.quantile(data, 0.1)


@numba.njit
def __stat_quantile_90(data: np.ndarray):
    """
    Compute the 90% quantile.

    :param data: the data
    :return: the 90% quantile
    """
    length: Final[int] = len(data)
    if (length > 10) and ((length % 10) == 1):
        data.sort()
        return data[(9 * (length - 1)) // 10]
    return np.quantile(data, 0.9)


@numba.njit
def __stat_quantile_159(data: np.ndarray):
    """
    Compute the 15.9% quantile, which equals mean-sd in normal distributions.

    :param data: the data
    :return: the 15.9% quantile
    """
    return np.quantile(data, __Q159)


@numba.njit
def __stat_quantile_841(data: np.ndarray):
    """
    Compute the 84.1% quantile, which equals mean+sd in normal distributions.

    :param data: the data
    :return: the 84.1% quantile
    """
    return np.quantile(data, __Q841)


#: The statistics key for the minimum
STAT_MINIMUM: Final[str] = statn.KEY_MINIMUM
#: The statistics key for the median.
STAT_MEDIAN: Final[str] = statn.KEY_MEDIAN
#: The statistics key for the arithmetic mean.
STAT_MEAN_ARITH: Final[str] = statn.KEY_MEAN_ARITH
#: The statistics key for the geometric mean.
STAT_MEAN_GEOM: Final[str] = statn.KEY_MEAN_GEOM
#: The statistics key for the maximum
STAT_MAXIMUM: Final[str] = statn.KEY_MAXIMUM
#: The statistics key for the standard deviation
STAT_STDDEV: Final[str] = statn.KEY_STDDEV
#: The key for the arithmetic mean minus the standard deviation.
STAT_MEAN_MINUS_STDDEV: Final[str] = f"{STAT_MEAN_ARITH}-{STAT_STDDEV}"
#: The key for the arithmetic mean plus the standard deviation.
STAT_MEAN_PLUS_STDDEV: Final[str] = f"{STAT_MEAN_ARITH}+{STAT_STDDEV}"
#: The key for the 10% quantile.
STAT_Q10: Final[str] = "q10"
#: The key for the 90% quantile.
STAT_Q90: Final[str] = "q90"
#: The key for the 15.9% quantile. In a normal distribution, this quantile
#: is where "mean - standard deviation" is located-
STAT_Q159: Final[str] = "q159"
#: The key for the 84.1% quantile. In a normal distribution, this quantile
#: is where "mean + standard deviation" is located-
STAT_Q841: Final[str] = "q841"

#: The internal function map.
_FUNC_MAP: Final[Dict[str, Callable]] = {
    STAT_MINIMUM: __stat_min,
    STAT_MEDIAN: __stat_median,
    STAT_MEAN_ARITH: __stat_arith_mean,
    STAT_MEAN_GEOM: __stat_geo_mean,
    STAT_MAXIMUM: __stat_max,
    STAT_STDDEV: __stat_sd,
    STAT_MEAN_MINUS_STDDEV: __stat_mean_minus_sd,
    STAT_MEAN_PLUS_STDDEV: __stat_mean_plus_sd,
    STAT_Q10: __stat_quantile_10,
    STAT_Q90: __stat_quantile_90,
    STAT_Q159: __stat_quantile_159,
    STAT_Q841: __stat_quantile_841
}


@dataclass(frozen=True, init=False, order=True)
class StatRun(MultiRun2DData):
    """A time-value statistic over a set of runs."""

    #: The name of this statistic.
    stat_name: str
    #: The time-dependent statistic.
    stat: np.ndarray

    def __init__(self,
                 algorithm: Optional[str],
                 instance: Optional[str],
                 n: int,
                 time_unit: str,
                 f_name: str,
                 stat_name: str,
                 stat: np.ndarray):
        """
        Create the time-based statistics of an algorithm-setup combination.

        :param algorithm: the algorithm name, if all runs are
            with the same algorithm
        :param instance: the instance name, if all runs are
            on the same instance
        :param n: the total number of runs
        :param time_unit: the time unit
        :param f_name: the objective dimension name
        :param stat_name: the name of the statistic
        :param stat: the statistic itself
        """
        super().__init__(algorithm, instance, n, time_unit, f_name)

        if not isinstance(stat_name, str):
            raise type_error(stat_name, "stat_name", str)
        object.__setattr__(self, "stat_name", stat_name)
        if not isinstance(stat, np.ndarray):
            raise type_error(stat, "statistic data", np.ndarray)
        stat.flags.writeable = False
        if (len(stat.shape) != 2) or (stat.shape[1] != 2) or \
                (stat.shape[0] <= 0):
            raise ValueError(
                "time array must be two-dimensional and have two columns and "
                f"at least one row, but has shape {stat.shape}.")
        if not is_np_float(stat.dtype):
            raise ValueError("statistics array must be float-typed, but has "
                             f"dtype {stat.dtype}.")
        object.__setattr__(self, "stat", stat)

    @staticmethod
    def create(source: Iterable[Progress],
               statistics: Union[str, Iterable[str]],
               consumer: Callable[['StatRun'], Any]) -> None:
        """
        Compute statistics from an iterable of `Progress` objects.

        :param source: the progress data
        :param statistics: the statistics to be computed
        :param consumer: the consumer for the statistics
        """
        if not isinstance(source, Iterable):
            raise type_error(source, "source", Iterable)
        if isinstance(statistics, str):
            statistics = [statistics]
        if not isinstance(statistics, Iterable):
            raise type_error(statistics, "statistics", Iterable)
        if not callable(consumer):
            raise type_error(consumer, "consumer", call=True)

        algorithm: Optional[str] = None
        instance: Optional[str] = None
        time_unit: Optional[str] = None
        f_name: Optional[str] = None
        time: List[np.ndarray] = []
        f: List[np.ndarray] = []
        n: int = 0

        for progress in source:
            if not isinstance(progress, Progress):
                raise type_error(progress, "stat run data source", Progress)
            if n <= 0:
                algorithm = progress.algorithm
                instance = progress.instance
                time_unit = progress.time_unit
                f_name = progress.f_name
            else:
                if algorithm != progress.algorithm:
                    algorithm = None
                if instance != progress.instance:
                    instance = None
                if time_unit != progress.time_unit:
                    raise ValueError(
                        f"Cannot mix time units {time_unit} "
                        f"and {progress.time_unit}.")
                if f_name != progress.f_name:
                    raise ValueError(f"Cannot mix f-names {f_name} "
                                     f"and {progress.f_name}.")
            n += 1
            time.append(progress.time)
            f.append(progress.f)

        if n <= 0:
            raise ValueError("Did not encounter any progress information.")

        x: Final[Tuple[np.ndarray, ...]] = tuple(time)
        del time
        y: Final[Tuple[np.ndarray, ...]] = tuple(f)
        del f

        x_unique = _unique_floats_1d(x)
        if not isinstance(x_unique, np.ndarray):
            raise type_error(x_unique, "x_unique", np.ndarray)
        if not is_np_float(x_unique.dtype):
            raise TypeError(
                f"x_unique must be floats, but is {x_unique.dtype}.")
        if (len(x_unique.shape) != 1) or (x_unique.shape[0] <= 0):
            raise ValueError(
                f"Invalid shape of unique values {x_unique.shape}.")

        count = 0
        for name in statistics:
            if not isinstance(name, str):
                raise type_error(name, "statistic name", str)
            if name not in _FUNC_MAP:
                raise ValueError(f"Unknown statistic name '{name}'.")
            consumer(StatRun(algorithm, instance, n, time_unit, f_name, name,
                             _apply_fun(x_unique, x, y, _FUNC_MAP[name])))
            count += 1

        if count <= 0:
            raise ValueError("No statistic names provided.")

    @staticmethod
    def from_progress(source: Iterable[Progress],
                      statistics: Union[str, Iterable[str]],
                      consumer: Callable[['StatRun'], Any],
                      join_all_algorithms: bool = False,
                      join_all_instances: bool = False) -> None:
        """
        Aggregate statist runs over a stream of progress data.

        :param source: the stream of progress data
        :param statistics: the statistics that should be computed per group
        :param consumer: the destination to which the new stat runs will be
            passed
        :param join_all_algorithms: should the statistics be aggregated
            over all algorithms
        :param join_all_instances: should the statistics be aggregated
            over all algorithms
        """
        if not isinstance(source, Iterable):
            raise type_error(source, "source", Iterable)
        if isinstance(statistics, str):
            statistics = [statistics]
        if not isinstance(statistics, Iterable):
            raise type_error(statistics, "statistics", Iterable)
        if not callable(consumer):
            raise type_error(consumer, "consumer", call=True)
        if not isinstance(join_all_algorithms, bool):
            raise type_error(join_all_algorithms, "join_all_algorithms", bool)
        if not isinstance(join_all_instances, bool):
            raise type_error(join_all_instances, "join_all_instances", bool)

        sorter: Dict[str, List[Progress]] = {}
        for prog in source:
            if not isinstance(prog, Progress):
                raise type_error(prog, "progress source", Progress)
            a: str = "" if join_all_algorithms else prog.algorithm
            i: str = "" if join_all_instances else prog.instance
            key: str = f"{a}/{i}/{prog.time_unit}/{prog.f_name}"
            if key in sorter:
                lst = sorter[key]
            else:
                lst = []
                sorter[key] = lst
            lst.append(prog)

        if len(sorter) <= 0:
            raise ValueError("source must not be empty")

        if len(sorter) > 1:
            keys = list(sorter.keys())
            keys.sort()
            for key in keys:
                StatRun.create(sorter[key], statistics, consumer)
        else:
            StatRun.create(next(iter(sorter.values())), statistics, consumer)


def get_statistic(obj: Union[PerRunData, MultiRunData]) -> Optional[str]:
    """
    Get the statistic of a given object.

    :param obj: the object
    :return: the statistic string, or `None` if no statistic is specified
    """
    return obj.stat_name if isinstance(obj, StatRun) else None
