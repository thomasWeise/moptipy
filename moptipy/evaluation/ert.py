"""
Approximate the expected running time to reach certain goals.

The (empirically estimated) Expected Running Time (ERT) tries to give an
impression of how long an algorithm needs to reach a certain solution quality.

The ERT for a problem instance is estimated as the ratio of the sum of all FEs
that all the runs consumed until they either have discovered a solution of a
given goal quality or exhausted their budget, divided by the number of runs
that discovered a solution of the goal quality. The ERT is the mean expect
runtime under the assumption of independent restarts after failed runs, which
then may either succeed (consuming the mean runtime of the successful runs) or
fail again (with the observed failure probability, after consuming the
available budget).

The ERT itself can be considered as a function that associates the estimated
runtime given above to all possible solution qualities that can be attained by
an algorithm for a give problem. For qualities/goals that an algorithm did not
attain in any run, the ERT becomes infinite.

1. Kenneth V. Price. Differential Evolution vs. The Functions of the 2nd ICEO.
   In Russ Eberhart, Peter Angeline, Thomas Back, Zbigniew Michalewicz, and
   Xin Yao, editors, *IEEE International Conference on Evolutionary
   Computation,* April 13-16, 1997, Indianapolis, IN, USA, pages 153-157.
   IEEE Computational Intelligence Society. ISBN: 0-7803-3949-5.
   doi: https://doi.org/10.1109/ICEC.1997.592287
2. Nikolaus Hansen, Anne Auger, Steffen Finck, Raymond Ros. *Real-Parameter
   Black-Box Optimization Benchmarking 2010: Experimental Setup.*
   Research Report RR-7215, INRIA. 2010. inria-00462481.
   https://hal.inria.fr/inria-00462481/document/
"""

from dataclasses import dataclass
from math import inf, isfinite
from typing import Any, Callable, Final, Iterable, cast

import numpy as np

import moptipy.api.logging as lg
import moptipy.utils.nputils as npu
from moptipy.evaluation._utils import _get_goal_reach_index
from moptipy.evaluation.base import (
    F_NAME_NORMALIZED,
    F_NAME_RAW,
    F_NAME_SCALED,
    KEY_N,
    MultiRun2DData,
)
from moptipy.evaluation.progress import Progress
from moptipy.utils.console import logger
from moptipy.utils.logger import (
    COMMENT_CHAR,
    CSV_SEPARATOR,
    KEY_VALUE_SEPARATOR,
)
from moptipy.utils.path import Path
from moptipy.utils.strings import num_to_str
from moptipy.utils.types import type_error


def compute_single_ert(source: Iterable[Progress],
                       goal_f: int | float) -> float:
    """
    Compute a single ERT.

    The ERT is the sum of the time that the runs spend with a
    best-so-far quality greater or equal than `goal_f` divided by the
    number of runs that reached `goal_f`. The idea is that the
    unsuccessful runs spent their complete computational budget and
    once they have terminated, we would immediately start a new,
    independent run.

    Warning: `source` must only contain progress objects that contain
    monotonously improving points. It must not contain runs that may get
    worse over time.

    :param source: the source array
    :param goal_f: the goal objective value
    :return: the ERT

    >>> from moptipy.evaluation.progress import Progress as Pr
    >>> from numpy import array as a
    >>> f = "plainF"
    >>> t = "FEs"
    >>> r = [Pr("a", "i", 1, a([1, 4, 8]), t, a([10, 8, 5]), f),
    ...      Pr("a", "i", 2, a([1, 3, 6]), t, a([9, 7, 4]), f),
    ...      Pr("a", "i", 3, a([1, 2, 7, 9]), t, a([8, 7, 6, 3]), f),
    ...      Pr("a", "i", 4, a([1, 12]), t, a([9, 3]), f)]
    >>> print(compute_single_ert(r, 11))
    1.0
    >>> print(compute_single_ert(r, 10))
    1.0
    >>> print(compute_single_ert(r, 9.5))  # (4 + 1 + 1 + 1) / 4 = 1.75
    1.75
    >>> print(compute_single_ert(r, 9))  # (4 + 1 + 1 + 1) / 4 = 1.75
    1.75
    >>> print(compute_single_ert(r, 8.5))  # (4 + 3 + 1 + 12) / 4 = 5
    5.0
    >>> print(compute_single_ert(r, 8))  # (4 + 3 + 1 + 12) / 4 = 5
    5.0
    >>> print(compute_single_ert(r, 7.3))  # (8 + 3 + 2 + 12) / 4 = 6.25
    6.25
    >>> print(compute_single_ert(r, 7))  # (8 + 3 + 2 + 12) / 4 = 6.25
    6.25
    >>> print(compute_single_ert(r, 6.1))  # (8 + 6 + 7 + 12) / 4 = 8.25
    8.25
    >>> print(compute_single_ert(r, 6))  # (8 + 6 + 7 + 12) / 4 = 8.25
    8.25
    >>> print(compute_single_ert(r, 5.7))  # (8 + 6 + 9 + 12) / 4 = 8.75
    8.75
    >>> print(compute_single_ert(r, 5))  # (8 + 6 + 9 + 12) / 4 = 8.75
    8.75
    >>> print(compute_single_ert(r, 4.2))  # (8 + 6 + 9 + 12) / 3 = 11.666...
    11.666666666666666
    >>> print(compute_single_ert(r, 4))  # (8 + 6 + 9 + 12) / 3 = 11.666...
    11.666666666666666
    >>> print(compute_single_ert(r, 3.8))  # (8 + 6 + 9 + 12) / 2 = 17.5
    17.5
    >>> print(compute_single_ert(r, 3))  # (8 + 6 + 9 + 12) / 2 = 17.5
    17.5
    >>> print(compute_single_ert(r, 2.9))
    inf
    >>> print(compute_single_ert(r, 2))
    inf
    """
    n_success: int = 0
    time_sum: int = 0
    for progress in source:
        idx = _get_goal_reach_index(progress.f, goal_f)
        if idx >= 0:
            n_success += 1
        else:
            idx = cast(np.integer, -1)
        time_sum = time_sum + int(progress.time[idx])
    if time_sum <= 0:
        raise ValueError(f"Time sum cannot be {time_sum}.")
    if n_success <= 0:
        return inf
    return time_sum / n_success


@dataclass(frozen=True, init=False, order=True)
class Ert(MultiRun2DData):
    """Estimate the Expected Running Time (ERT)."""

    #: The ert function
    ert: np.ndarray

    def __init__(self,
                 algorithm: str | None,
                 instance: str | None,
                 n: int,
                 time_unit: str,
                 f_name: str,
                 ert: np.ndarray):
        """
        Create the Ert function.

        :param algorithm: the algorithm name, if all runs are with the same
            algorithm
        :param instance: the instance name, if all runs are on the same
            instance
        :param n: the total number of runs
        :param time_unit: the time unit
        :param f_name: the objective dimension name
        :param ert: the ert matrix
        """
        super().__init__(algorithm, instance, n, time_unit, f_name)
        if not isinstance(ert, np.ndarray):
            raise type_error(ert, "ert", np.ndarray)
        ert.flags.writeable = False

        f: Final[np.ndarray] = ert[:, 0]
        if not npu.is_all_finite(f):
            raise ValueError(
                f"Ert x-axis must be all finite, but encountered {f}.")
        ll = f.size
        if (ll > 1) and np.any(f[1:] <= f[:-1]):
            raise ValueError("f data must be strictly increasing,"
                             f"but encountered {f}.")

        t: Final[np.ndarray] = ert[:, 1]
        if np.isfinite(t[0]) or (np.isposinf(t[0])):
            if not npu.is_all_finite(t[1:]):
                raise ValueError(
                    "non-first ert y-axis elements must be all finite.")
            if np.any(t[1:] >= t[:-1]):
                raise ValueError("t data must be strictly decreasing,"
                                 f"but encountered {t}.")
        else:
            raise ValueError(
                f"first ert y-axis element cannot be {t[0]}.")
        object.__setattr__(self, "ert", ert)

    def to_csv(self, file: str,
               put_header: bool = True) -> Path:
        """
        Store a :class:`Ert` record in a CSV file.

        :param file: the file to generate
        :param put_header: should we put a header with meta-data?
        :return: the fully resolved file name
        """
        path: Final[Path] = Path.path(file)
        logger(f"Writing ERT to CSV file {path!r}.")

        with path.open_for_write() as out:
            sep: Final[str] = CSV_SEPARATOR
            if put_header:
                kv: Final[str] = KEY_VALUE_SEPARATOR
                cmt: Final[str] = COMMENT_CHAR
                if self.algorithm is not None:
                    out.write(
                        f"{cmt} {lg.KEY_ALGORITHM}{kv}{self.algorithm}\n")
                if self.instance is not None:
                    out.write(
                        f"{cmt} {lg.KEY_INSTANCE}{kv}{self.instance}\n")
                out.write(
                    f"{cmt} {KEY_N}{kv}{self.n}\n")
            out.write(f"{self.f_name}{sep}ert[{self.time_unit}]\n")
            for v in self.ert:
                out.write(
                    f"{num_to_str(v[0])}{sep}{num_to_str(v[1])}\n")

        logger(f"Done writing ERT to CSV file {path!r}.")

        path.enforce_file()
        return path

    @staticmethod
    def create(source: Iterable[Progress],
               f_lower_bound: int | float | Callable | None = None,
               use_default_lower_bounds: bool = True) -> "Ert":
        """
        Create one single Ert record from an iterable of Progress records.

        :param source: the set of progress instances
        :param f_lower_bound: the lower bound for the objective value, or a
            callable that is applied to a progress object to get the lower
            bound
        :param use_default_lower_bounds: should we use the default lower
            bounds
        :return: the Ert record
        """
        if not isinstance(source, Iterable):
            raise type_error(source, "source", Iterable)

        lower_bound: int | float = inf
        if (f_lower_bound is not None) and (not callable(f_lower_bound)):
            if not isfinite(f_lower_bound):
                raise ValueError("f_lower_bound must be finite "
                                 f"but is {f_lower_bound}.")
            lower_bound = f_lower_bound
            f_lower_bound = None

        algorithm: str | None = None
        instance: str | None = None
        time_unit: str | None = None
        f_name: str | None = None
        f_list: list[np.ndarray] = []
        n: int = 0

        prgs: Final[list[Progress]] = cast(list[Progress], source) \
            if isinstance(source, list) else list(source)

        for progress in prgs:
            if not isinstance(progress, Progress):
                raise type_error(progress, "progress input", Progress)
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
            if use_default_lower_bounds and \
                    (progress.f_standard is not None) and \
                    (progress.f_name == F_NAME_RAW) and \
                    (lower_bound > progress.f_standard):
                lower_bound = progress.f_standard
            if f_lower_bound is not None:
                lb = f_lower_bound(progress)
                if not isinstance(lb, int | float):
                    raise type_error(lb, "computed lower bound", (int, float))
                if not isfinite(lb):
                    raise ValueError(f"Invalid computed lower bound {lb}.")
                if lb < lower_bound:
                    lower_bound = lb
            f_list.append(progress.f)

        if n <= 0:
            raise ValueError("Did not encounter any progress information.")

        if use_default_lower_bounds:
            if f_name == F_NAME_SCALED:
                lower_bound = min(lower_bound, 1)
            elif f_name == F_NAME_NORMALIZED:
                lower_bound = min(lower_bound, 0)

        # get unique x-values and make sure that lower bound is included
        has_lb: Final[bool] = isfinite(lower_bound)
        if has_lb:
            f_list.append(np.array([lower_bound]))
        x = np.concatenate(f_list)
        del f_list
        x = np.unique(x)
        if has_lb:
            x = x[x >= lower_bound]
        base_len: Final[int] = x.size

        # prepare for backward iteration over arrays
        indices = np.array([(ppr.f.size - 1) for ppr in prgs],
                           dtype=npu.DEFAULT_INT)
        y = np.empty(base_len, dtype=npu.DEFAULT_FLOAT)

        for out_index in range(base_len):
            f_lim = x[out_index]
            found = 0
            time_sum: int = 0
            for r_idx, pr in enumerate(prgs):
                idx = indices[r_idx]
                f_vals = pr.f
                # did we fulfill the limit?
                if f_vals[idx] <= f_lim:
                    found += 1
                    # can we move time cursor back?
                    if (idx > 0) and (f_vals[idx - 1] <= f_lim):
                        idx -= 1
                        indices[r_idx] = idx
                else:
                    # condition not fulfilled, need to use maximum time
                    idx = -1
                time_sum = time_sum + int(pr.time[idx])

            # compute ert value: infinite if no run meets condition
            y[out_index] = inf if (found <= 0) else time_sum / found

        # convert the two arrays into one matrix
        ert = np.concatenate((x, y)).reshape((base_len, 2), order="F")

        return Ert(algorithm, instance, n,
                   time_unit, f_name, ert)

    @staticmethod
    def from_progresses(source: Iterable[Progress],
                        consumer: Callable[["Ert"], Any],
                        f_lower_bound: float | None = None,
                        use_default_lower_bounds: bool = True,
                        join_all_algorithms: bool = False,
                        join_all_instances: bool = False) -> None:
        """
        Compute one or multiple ERTs from a stream of end results.

        :param source: the set of progress instances
        :param f_lower_bound: the lower bound for the objective value
        :param use_default_lower_bounds: should we use the default lower
            bounds
        :param consumer: the destination to which the new records will be
            passed, can be the `append` method of a :class:`list`
        :param join_all_algorithms: should the Ert be aggregated over all
            algorithms
        :param join_all_instances: should the Ert be aggregated over all
            algorithms
        """
        if not isinstance(source, Iterable):
            raise type_error(source, "source", Iterable)
        if not callable(consumer):
            raise type_error(consumer, "consumer", call=True)
        if not isinstance(join_all_algorithms, bool):
            raise type_error(join_all_algorithms, "join_all_algorithms", bool)
        if not isinstance(join_all_instances, bool):
            raise type_error(join_all_instances, "join_all_instances", bool)

        if join_all_algorithms and join_all_instances:
            consumer(Ert.create(source, f_lower_bound,
                                use_default_lower_bounds))
            return

        sorter: dict[str, list[Progress]] = {}
        for er in source:
            if not isinstance(er, Progress):
                raise type_error(er, "progress source", Progress)
            key = er.instance if join_all_algorithms else \
                er.algorithm if join_all_instances else \
                f"{er.algorithm}/{er.instance}"
            if key in sorter:
                lst = sorter[key]
            else:
                lst = []
                sorter[key] = lst
            lst.append(er)

        if len(sorter) <= 0:
            raise ValueError("source must not be empty")

        if len(sorter) > 1:
            keyz = list(sorter.keys())
            keyz.sort()
            for key in keyz:
                consumer(Ert.create(sorter[key], f_lower_bound,
                                    use_default_lower_bounds))
        else:
            consumer(Ert.create(next(iter(sorter.values())), f_lower_bound,
                                use_default_lower_bounds))
