"""Approximate the ECDF over the ERT to reach certain goals."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite, inf
from typing import Optional, Iterable, List, Final, Union, \
    Dict, Callable

import numpy as np

import moptipy.utils.logging as lg
import moptipy.utils.nputils as npu
from moptipy.evaluation.base import MultiRun2DData, F_NAME_SCALED, \
    F_NAME_NORMALIZED, KEY_N
from moptipy.evaluation.ert import compute_single_ert
from moptipy.evaluation.progress import Progress
from moptipy.utils.io import canonicalize_path, enforce_file

#: The number of instances.
KEY_N_INSTS: Final[str] = f"{KEY_N}Insts"
#: The objective dimension name.
KEY_F_NAME: Final[str] = "fName"


@dataclass(frozen=True, init=False, order=True)
class ErtEcdf(MultiRun2DData):
    """The ERT-ECDF."""

    #: The ECDF over the ERT function
    ert_ecdf: np.ndarray
    #: The number of instances over which the ERT-ECDF is computed.
    n_insts: int
    #: The goal value, or None if different goals were used for different
    #   instances
    goal_f: Union[int, float, None]

    def __init__(self,
                 algorithm: Optional[str],
                 n: int,
                 n_insts: int,
                 time_unit: str,
                 f_name: str,
                 goal_f: Union[int, float, None],
                 ert_ecdf: np.ndarray):
        """
        Create the ERT-ECDF function.

        :param Optional[str] algorithm: the algorithm name, if all runs are
            with the same algorithm
        :param int n: the total number of runs
        :param int n_insts: the total number of instances
        :param str time_unit: the time unit
        :param str f_name: the objective dimension name
        :param Union[int, float, None] goal_f: the goal value, or None if
            different goals were used for different instances
        :param np.ndarray ert_ecdf: the ert-ecdf matrix
        """
        super().__init__(algorithm, None, n, time_unit, f_name)

        if not isinstance(n_insts, int):
            raise TypeError(f"n_insts must be int but is {type(n_insts)}.")
        if (n_insts < 1) or (n_insts > self.n):
            raise ValueError("n_insts must be > 0 and < n_runs "
                             f"({self.n}), but got {n_insts}.")
        object.__setattr__(self, "n_insts", n_insts)

        if goal_f is not None:
            if not isinstance(goal_f, (int, float)):
                raise TypeError(f"Invalid goal_f type {type(goal_f)}.")
            if not isfinite(goal_f):
                raise ValueError(f"Invalid goal_f {goal_f}.")
        object.__setattr__(self, "goal_f", goal_f)

        if not isinstance(ert_ecdf, np.ndarray):
            raise TypeError(
                f"ert-ecdf must be numpy.ndarray, but is {type(ert_ecdf)}.")
        ert_ecdf.flags.writeable = False

        time: Final[np.ndarray] = ert_ecdf[:, 0]
        ll = time.size
        if ll < 2:
            raise ValueError("Must have at least two points in "
                             f"ERT-ECDF curve , but encountered {ll}.")
        if not npu.is_all_finite(time[:-1]):
            raise ValueError("Non-last Ert-based time-values must be finite, "
                             f"but encountered {time}.")
        if np.any(time[1:] <= time[:-1]):
            raise ValueError("ert time data must be strictly increasing,"
                             f"but encountered {time}.")

        prb: Final[np.ndarray] = ert_ecdf[:, 1]
        if not npu.is_all_finite(prb):
            raise ValueError(
                f"All ECDF values must be finite, but encountered {prb}.")
        if ll > 2:
            if np.any(prb[1:-1] <= prb[:-2]):
                raise ValueError("ECDF data must be strictly increasing,"
                                 f"but encountered {prb}.")
        if prb[0] < 0:
            raise ValueError(f"First ECDF element cannot be {prb[0]}.")
        if prb[ll - 1] > 1:
            raise ValueError(f"Last ECDF element cannot be {prb[ll - 1]}.")
        object.__setattr__(self, "ert_ecdf", ert_ecdf)

    def to_csv(self, file: str,
               put_header: bool = True) -> str:
        """
        Store a :class:`ErtEcdf` record in a CSV file.

        :param str file: the file to generate
        :param bool put_header: should we put a header with meta-data?
        :return: the fully resolved file name
        :rtype: str
        """
        file = canonicalize_path(file)
        print(f"{datetime.now()}: Writing ERT-ECDF to CSV file '{file}'.")

        with open(file, "wt") as out:
            sep: Final[str] = lg.CSV_SEPARATOR
            if put_header:
                kv: Final[str] = lg.KEY_VALUE_SEPARATOR
                cmt: Final[str] = lg.COMMENT_CHAR
                if self.algorithm is not None:
                    out.write(
                        f"{cmt} {lg.KEY_ALGORITHM}{kv}{self.algorithm}\n")
                out.write(
                    f"{cmt} {KEY_N}{kv}{self.n}\n")
                out.write(
                    f"{cmt} {KEY_N_INSTS}{kv}{self.n_insts}\n")
                out.write(
                    f"{cmt} {KEY_F_NAME}{kv}{self.f_name}\n")
                if self.goal_f is not None:
                    out.write(
                        f"{cmt} {lg.KEY_GOAL_F}{kv}{self.goal_f}\n")

            out.write(f"ert[{self.time_unit}]{sep}ecdf\n")
            for v in self.ert_ecdf:
                out.write(
                    f"{lg.num_to_str(v[0])}{sep}{lg.num_to_str(v[1])}\n")

        print(
            f"{datetime.now()}: Done writing ERT-ECDF to CSV file '{file}'.")

        return enforce_file(file)

    @staticmethod
    def create(source: Iterable[Progress],
               goal_f: Union[int, float, Callable, None] = None,
               use_default_goal_f: bool = True) -> 'ErtEcdf':
        """
        Create one single Ert-Ecdf record from an iterable of Progress records.

        :param Iterable[moptipy.evaluation.Progress] source: the set of
            progress instances
        :param Union[int,float,Callable, None] goal_f: the goal objective
            value
        :param bool use_default_goal_f: should we use the default lower
            bounds as goals?
        :return: the Ert record
        :rtype: ErtEcdf
        """
        if not isinstance(source, Iterable):
            raise TypeError(
                f"source must be Iterable, but is {type(source)}.")

        algorithm: Optional[str] = None
        time_unit: Optional[str] = None
        f_name: Optional[str] = None
        inst_runs: Dict[str, List[Progress]] = dict()
        n: int = 0

        for progress in source:
            if not isinstance(progress, Progress):
                raise TypeError("Only Progress records are permitted, but "
                                f"encountered a {type(progress)}.")
            if n <= 0:
                algorithm = progress.algorithm
                time_unit = progress.time_unit
                f_name = progress.f_name
            else:
                if algorithm != progress.algorithm:
                    algorithm = None
                if time_unit != progress.time_unit:
                    raise ValueError(
                        f"Cannot mix time units {time_unit} "
                        f"and {progress.time_unit}.")
                if f_name != progress.f_name:
                    raise ValueError(f"Cannot mix f-names {f_name} "
                                     f"and {progress.f_name}.")
            n += 1
            if progress.instance in inst_runs:
                inst_runs[progress.instance].append(progress)
            else:
                inst_runs[progress.instance] = [progress]
        del source

        if n <= 0:
            raise ValueError("Did not encounter any progress information.")
        n_insts: Final[int] = len(inst_runs)
        if (n_insts <= 0) or (n_insts > n):
            raise ValueError("Huh?.")

        erts: List[float] = list()
        goal: Union[int, float, None]
        same_goal_f: Union[int, float, None] = None
        first: bool = True
        for instance, pl in inst_runs.items():
            goal = None
            if isinstance(goal_f, (int, float)):
                goal = goal_f
            elif callable(goal_f):
                goal = goal_f(instance)
            if (goal is None) and use_default_goal_f:
                if f_name == F_NAME_SCALED:
                    goal = 1
                elif f_name == F_NAME_NORMALIZED:
                    goal = 0
                else:
                    goal = pl[0].f_standard
                    for pp in pl:
                        if goal != pp.f_standard:
                            raise ValueError("Inconsistent goals: "
                                             f"{goal} and {pp.f_standard}")
            if not isinstance(goal, (int, float)):
                raise TypeError(
                    f"Goal must be int or float but is {type(goal)}.")
            if first:
                same_goal_f = goal
            elif goal != same_goal_f:
                same_goal_f = None
            t = compute_single_ert(pl, goal)
            if isfinite(t):
                if t < 0:
                    raise ValueError(f"Invalid ert {t}.")
                erts.append(t)
            elif not (t >= inf):
                raise ValueError(f"Invalid ert {t}.")
        del inst_runs

        if len(erts) <= 0:
            return ErtEcdf(algorithm, n, n_insts, time_unit,
                           f_name, same_goal_f,
                           np.array([[0, 0], [inf, 0]]))

        erts.sort()
        time: List[float] = [0]
        ecdf: List[float] = [0]
        success: int = 0
        ll: int = 0
        for t in erts:
            success += 1
            if t > time[ll]:
                time.append(t)
                ecdf.append(success / n_insts)
                ll += 1
            else:
                ecdf[ll] = success / n_insts

        time.append(inf)
        ecdf.append(ecdf[ll])

        return ErtEcdf(algorithm, n, n_insts,
                       time_unit, f_name,
                       same_goal_f,
                       np.column_stack((np.array(time),
                                        np.array(ecdf))))
