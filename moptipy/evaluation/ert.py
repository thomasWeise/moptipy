"""Approximate the empirical running time to reach certain goals."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite, inf
from typing import Optional, Iterable, List, Final, cast, Union

import numba  # type: ignore
import numpy as np

import moptipy.utils.logging as lg
import moptipy.utils.nputils as npu
from moptipy.evaluation.base import MultiRun2DData, F_NAME_SCALED, \
    F_NAME_NORMALIZED, F_NAME_RAW, KEY_N
from moptipy.evaluation.progress import Progress
from moptipy.utils.io import canonicalize_path, enforce_file


@numba.njit(nogil=True)
def __get_ert_index(f: np.ndarray, goal_f: float) -> np.integer:
    """
    Compute the ert index.

    :param np.ndarray f: the raw data
    :param float goal_f: the goal f value
    :return: the index
    :rtype: np.integer
    """
    return np.searchsorted(f[::-1], goal_f, side="right")


def compute_single_ert(source: Iterable[Progress],
                       goal_f: float) -> float:
    """
    Compute a single ERT.

    :param Iterable[moptipy.evaluation.Progress] source: the source array
    :param float goal_f: the goal objective value
    :return: the ERT
    :rtype: float
    """
    n_success: int = 0
    time_sum: int = 0
    for progress in source:
        size = progress.f.size
        idx = size - __get_ert_index(progress.f, goal_f)
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
    """A time-value statistic over a set of runs."""

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
        if not npu.is_all_finite(ert[:, 0]):
            raise ValueError("ert x-axis must be all finite.")
        if np.isfinite(ert[0, 1]) or (np.isposinf(ert[0, 1])):
            if not npu.is_all_finite(ert[1:, 1]):
                raise ValueError(
                    "non-first ert y-axis elements must be all finite.")
        else:
            raise ValueError(
                f"first ert y-axis element cannot be {ert[0, 1]}.")
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
    def from_progress(source: Iterable[Progress],
                      f_lower_bound: Optional[float] = None,
                      use_default_lower_bounds: bool = True) -> 'Ert':
        """
        Create an Ert record.

        :param Iterable[moptipy.evaluation.Progress] source: the set of
            progress instances
        :param float f_lower_bound: the lower bound for the objective value
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
            f_lower_bound = float(f_lower_bound)
            if not isfinite(f_lower_bound):
                raise ValueError(
                    f"f_lower_bound must be finite but is {f_lower_bound}.")
            lower_bound = f_lower_bound

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
            f_list.append(progress.f)

        if n <= 0:
            raise ValueError("Did not encounter any progress information.")

        if use_default_lower_bounds:
            if f_name == F_NAME_SCALED:
                if lower_bound >= 1:
                    lower_bound = 1
            elif f_name == F_NAME_NORMALIZED:
                if lower_bound >= 0:
                    lower_bound = 0

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
