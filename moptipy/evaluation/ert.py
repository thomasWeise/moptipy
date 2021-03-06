"""Approximate the expected running time to reach certain goals."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite, inf
from typing import Optional, Iterable, List, Final, cast, Union, \
    MutableSequence, Dict, Callable

import numpy as np

import moptipy.utils.logging as lg
import moptipy.utils.nputils as npu
from moptipy.evaluation._utils import _get_reach_index
from moptipy.evaluation.base import MultiRun2DData, F_NAME_SCALED, \
    F_NAME_NORMALIZED, F_NAME_RAW, KEY_N
from moptipy.evaluation.progress import Progress
from moptipy.utils.io import canonicalize_path, enforce_file


def compute_single_ert(source: Iterable[Progress],
                       goal_f: Union[int, float]) -> float:
    """
    Compute a single ERT.

    :param Iterable[moptipy.evaluation.Progress] source: the source array
    :param Union[int, float] goal_f: the goal objective value
    :return: the ERT
    :rtype: float
    """
    n_success: int = 0
    time_sum: int = 0
    for progress in source:
        size = progress.f.size
        idx = size - _get_reach_index(progress.f, goal_f)
        if idx < size:
            n_success += 1
        else:
            idx = cast(np.integer, -1)
        time_sum += progress.time[idx]
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
                 algorithm: Optional[str],
                 instance: Optional[str],
                 n: int,
                 time_unit: str,
                 f_name: str,
                 ert: np.ndarray):
        """
        Create the Ert function.

        :param Optional[str] algorithm: the algorithm name, if all runs are
            with the same algorithm
        :param Optional[str] instance: the instance name, if all runs are
            on the same instance
        :param int n: the total number of runs
        :param str time_unit: the time unit
        :param str f_name: the objective dimension name
        :param np.ndarray ert: the ert matrix
        """
        super().__init__(algorithm, instance, n, time_unit, f_name)
        if not isinstance(ert, np.ndarray):
            raise TypeError(f"ert must be numpy.ndarray, but is {type(ert)}.")
        ert.flags.writeable = False

        f: Final[np.ndarray] = ert[:, 0]
        if not npu.is_all_finite(f):
            raise ValueError(
                f"Ert x-axis must be all finite, but encountered {f}.")
        ll = f.size
        if ll > 1:
            if np.any(f[1:] <= f[:-1]):
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
               put_header: bool = True) -> str:
        """
        Store a :class:`Ert` record in a CSV file.

        :param str file: the file to generate
        :param bool put_header: should we put a header with meta-data?
        :return: the fully resolved file name
        :rtype: str
        """
        file = canonicalize_path(file)
        print(f"{datetime.now()}: Writing ERT to CSV file '{file}'.")

        with open(file, "wt") as out:
            sep: Final[str] = lg.CSV_SEPARATOR
            if put_header:
                kv: Final[str] = lg.KEY_VALUE_SEPARATOR
                cmt: Final[str] = lg.COMMENT_CHAR
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
                    f"{lg.num_to_str(v[0])}{sep}{lg.num_to_str(v[1])}\n")

        print(f"{datetime.now()}: Done writing ERT to CSV file '{file}'.")

        return enforce_file(file)

    @staticmethod
    def create(source: Iterable[Progress],
               f_lower_bound: Union[int, float, Callable, None] = None,
               use_default_lower_bounds: bool = True) -> 'Ert':
        """
        Create one single Ert record from an iterable of Progress records.

        :param Iterable[moptipy.evaluation.Progress] source: the set of
            progress instances
        :param Union[int,float,Callable,None] f_lower_bound: the lower bound
            for the objective value, or a callable that is applied to a
            progress object to get the lower bound
        :param bool use_default_lower_bounds: should we use the default lower
            bounds
        :return: the Ert record
        :rtype: Ert
        """
        if not isinstance(source, Iterable):
            raise TypeError(
                f"source must be Iterable, but is {type(source)}.")

        lower_bound: Union[int, float] = inf
        if f_lower_bound is not None:
            if not callable(f_lower_bound):
                if not isfinite(f_lower_bound):
                    raise ValueError("f_lower_bound must be finite "
                                     f"but is {f_lower_bound}.")
                lower_bound = f_lower_bound
                f_lower_bound = None

        algorithm: Optional[str] = None
        instance: Optional[str] = None
        time_unit: Optional[str] = None
        f_name: Optional[str] = None
        f_list: List[np.ndarray] = list()
        n: int = 0

        prgs: Final[List[Progress]] = cast(List[Progress], source) \
            if isinstance(source, list) else list(source)

        for progress in prgs:
            if not isinstance(progress, Progress):
                raise TypeError("Only Progress records are permitted, but "
                                f"encountered a {type(progress)}.")
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
                    (progress.f_name == F_NAME_RAW):
                if lower_bound > progress.f_standard:
                    lower_bound = progress.f_standard
            if f_lower_bound is not None:
                lb = f_lower_bound(progress)
                if not isinstance(lb, (int, float)):
                    raise TypeError("Computed lower bound must be int or "
                                    f"float, but is {type(lb)}.")
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
                time_sum += pr.time[idx]

            # compute ert value: infinite if no run meets condition
            y[out_index] = inf if (found <= 0) else time_sum / found

        # convert the two arrays into one matrix
        ert = np.concatenate((x, y)).reshape((base_len, 2), order='F')

        return Ert(algorithm, instance, n,
                   time_unit, f_name, ert)

    @staticmethod
    def from_progresses(source: Iterable[Progress],
                        collector: MutableSequence['Ert'],
                        f_lower_bound: Optional[float] = None,
                        use_default_lower_bounds: bool = True,
                        join_all_algorithms: bool = False,
                        join_all_instances: bool = False) -> None:
        """
        Compute one or multiple ERTs from a stream of end results.

        :param Iterable[moptipy.evaluation.Progress] source: the set of
            progress instances
        :param float f_lower_bound: the lower bound for the objective value
        :param bool use_default_lower_bounds: should we use the default lower
            bounds
        :param MutableSequence['Ert'] collector: the destination
            to which the new records will be appended
        :param bool join_all_algorithms: should the Ert be aggregated over
            all algorithms
        :param bool join_all_instances: should the Ert be aggregated over all
            algorithms
        """
        if not isinstance(source, Iterable):
            raise TypeError(
                f"source must be Iterable, but is {type(source)}.")
        if not isinstance(collector, MutableSequence):
            raise TypeError("collector must be MutableSequence, "
                            f"but is {type(collector)}.")
        if not isinstance(join_all_algorithms, bool):
            raise TypeError("join_all_algorithms must be bool, "
                            f"but is {type(join_all_algorithms)}.")
        if not isinstance(join_all_instances, bool):
            raise TypeError("join_all_instances must be bool, "
                            f"but is {type(join_all_instances)}.")

        if join_all_algorithms and join_all_instances:
            collector.append(Ert.create(source, f_lower_bound,
                                        use_default_lower_bounds))
            return

        sorter: Dict[str, List[Progress]] = dict()
        for er in source:
            if not isinstance(er, Progress):
                raise TypeError("source must contain only Progress, but "
                                f"found a {type(er)}.")
            key = er.instance if join_all_algorithms else \
                er.algorithm if join_all_instances else \
                f"{er.algorithm}/{er.instance}"
            if key in sorter:
                lst = sorter[key]
            else:
                lst = list()
                sorter[key] = lst
            lst.append(er)

        if len(sorter) <= 0:
            raise ValueError("source must not be empty")

        if len(sorter) > 1:
            keyz = list(sorter.keys())
            keyz.sort()
            for key in keyz:
                collector.append(Ert.create(sorter[key], f_lower_bound,
                                            use_default_lower_bounds))
        else:
            collector.append(Ert.create(next(iter(sorter.values())),
                                        f_lower_bound,
                                        use_default_lower_bounds))
