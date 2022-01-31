"""Approximate the ECDF to reach certain goals."""

from dataclasses import dataclass
from math import isfinite, inf
from typing import Optional, Iterable, List, Final, Union, \
    Dict, Callable, MutableSequence

import numpy as np

import moptipy.utils.logging as lg
import moptipy.utils.nputils as npu
from moptipy.evaluation._utils import _get_reach_index
from moptipy.evaluation.base import MultiRun2DData, F_NAME_SCALED, \
    F_NAME_NORMALIZED, KEY_N
from moptipy.evaluation.lang import Lang
from moptipy.evaluation.progress import Progress
from moptipy.utils.log import logger
from moptipy.utils.path import Path

#: The number of instances.
KEY_N_INSTS: Final[str] = f"{KEY_N}Insts"
#: The objective dimension name.
KEY_F_NAME: Final[str] = "fName"


@dataclass(frozen=True, init=False, order=True)
class Ecdf(MultiRun2DData):
    """The ECDF."""

    #: The ECDF data function
    ecdf: np.ndarray
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
                 ecdf: np.ndarray):
        """
        Create the ECDF function.

        :param Optional[str] algorithm: the algorithm name, if all runs are
            with the same algorithm
        :param int n: the total number of runs
        :param int n_insts: the total number of instances
        :param str time_unit: the time unit
        :param str f_name: the objective dimension name
        :param Union[int, float, None] goal_f: the goal value, or None if
            different goals were used for different instances
        :param np.ndarray ecdf: the ert-ecdf matrix
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

        if not isinstance(ecdf, np.ndarray):
            raise TypeError(
                f"ecdf must be numpy.ndarray, but is {type(ecdf)}.")
        ecdf.flags.writeable = False

        time: Final[np.ndarray] = ecdf[:, 0]
        ll = time.size
        if ll < 2:
            raise ValueError("Must have at least two points in "
                             f"ecdf curve , but encountered {ll}.")
        if not npu.is_all_finite(time[:-1]):
            raise ValueError("Non-last Ert-based time-values must be finite, "
                             f"but encountered {time}.")
        if np.any(time[1:] <= time[:-1]):
            raise ValueError("Time data must be strictly increasing,"
                             f"but encountered {time}.")

        prb: Final[np.ndarray] = ecdf[:, 1]
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
        object.__setattr__(self, "ecdf", ecdf)

    def time_label(self) -> str:
        """
        Get the time label for x-axes.

        :return: the time key
        :rtype: str
        """
        return Lang.translate(self.time_unit)

    def _time_key(self) -> str:
        """
        Get the time key.

        :return: the time key
        :rtype: str
        """
        return self.time_unit

    def to_csv(self, file: str,
               put_header: bool = True) -> Path:
        """
        Store a :class:`Ecdf` record in a CSV file.

        :param str file: the file to generate
        :param bool put_header: should we put a header with meta-data?
        :return: the fully resolved file name
        :rtype: Path
        """
        path: Final[Path] = Path.path(file)
        logger(f"Writing ECDF to CSV file '{path}'.")

        with path.open_for_write() as out:
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

            out.write(f"{self._time_key()}{sep}ecdf\n")
            for v in self.ecdf:
                out.write(
                    f"{lg.num_to_str(v[0])}{sep}{lg.num_to_str(v[1])}\n")

        logger(f"Done writing ECDF to CSV file '{path}'.")

        path.enforce_file()
        return path

    @staticmethod
    def _compute_times(source: List[Progress],
                       goal: Union[int, float]) -> List[float]:
        """
        Compute the times for the given goals.

        :param source: the source array
        :param goal: the goal value
        :return: a list of times
        :rtype: List[float]
        """
        ret = []
        for pr in source:
            idx = _get_reach_index(pr.f, goal)
            if idx <= 0:
                continue
            ret.append(pr.time[pr.time.size - idx])
        return ret

    # noinspection PyUnusedLocal
    @staticmethod
    def _get_div(n: int, n_insts: int) -> int:
        """
        Get the divisor.

        :param int n: the number of runs
        :param int n_insts: the number of instances
        :return: the divisor
        :rtype: int
        """
        del n_insts
        return n

    @classmethod
    def create(cls,
               source: Iterable[Progress],
               goal_f: Union[int, float, Callable, None] = None,
               use_default_goal_f: bool = True) -> 'Ecdf':
        """
        Create one single Ecdf record from an iterable of Progress records.

        :param Iterable[moptipy.evaluation.Progress] source: the set of
            progress instances
        :param Union[int,float,Callable, None] goal_f: the goal objective
            value
        :param bool use_default_goal_f: should we use the default lower
            bounds as goals?
        :return: the Ecdf record
        :rtype: Ecdf
        """
        if not isinstance(source, Iterable):
            raise TypeError(
                f"source must be Iterable, but is {type(source)}.")

        algorithm: Optional[str] = None
        time_unit: Optional[str] = None
        f_name: Optional[str] = None
        inst_runs: Dict[str, List[Progress]] = {}
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

        times: List[float] = []
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
            res = cls._compute_times(pl, goal)
            for t in res:
                if isfinite(t):
                    if t < 0:
                        raise ValueError(f"Invalid ert {t}.")
                    times.append(t)
                elif not (t >= inf):
                    raise ValueError(f"Invalid ert {t}.")
        del inst_runs

        if len(times) <= 0:
            return cls(algorithm, n, n_insts, time_unit,
                       f_name, same_goal_f,
                       np.array([[0, 0], [inf, 0]]))

        times.sort()
        time: List[float] = [0]
        ecdf: List[float] = [0]
        success: int = 0
        div: Final[int] = cls._get_div(n, n_insts)
        ll: int = 0
        for t in times:
            success += 1
            if t > time[ll]:
                time.append(t)
                ecdf.append(success / div)
                ll += 1
            else:
                ecdf[ll] = success / div

        time.append(inf)
        ecdf.append(ecdf[ll])

        return cls(algorithm, n, n_insts,
                   time_unit, f_name,
                   same_goal_f,
                   np.column_stack((np.array(time),
                                    np.array(ecdf))))

    @classmethod
    def from_progresses(cls,
                        source: Iterable[Progress],
                        collector: MutableSequence['Ecdf'],
                        f_goal: Union[int, float, Callable,
                                      Iterable[Union[int, float,
                                                     Callable]]] = None,
                        join_all_algorithms: bool = False) -> None:
        """
        Compute one or multiple ECDFs from a stream of end results.

        :param Iterable[moptipy.evaluation.Progress] source: the set of
            progress instances
        :param f_goal: one or multiple goal values
        :param MutableSequence['Ert'] collector: the destination
            to which the new records will be appended
        :param bool join_all_algorithms: should the Ert-Ecdf be aggregated
            over all algorithms
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
        if not isinstance(f_goal, Iterable):
            f_goal = [f_goal]

        sorter: Dict[str, List[Progress]] = {}
        for er in source:
            if not isinstance(er, Progress):
                raise TypeError("source must contain only Progress, but "
                                f"found a {type(er)}.")
            key = er.algorithm
            if key in sorter:
                lst = sorter[key]
            else:
                lst = []
                sorter[key] = lst
            lst.append(er)

        if len(sorter) <= 0:
            raise ValueError("source must not be empty")

        keyz = list(sorter.keys())
        keyz.sort()

        for goal in f_goal:
            use_default_goal = goal is None
            for key in keyz:
                collector.append(cls.create(sorter[key],
                                            goal,
                                            use_default_goal))


def get_goal(ecdf: Ecdf) -> Union[int, float, None]:
    """
    Get the goal value from the given ecdf instance.

    :param Ecdf ecdf: the ecdf instance
    :return: the goal value
    :rtype: Union[int, float, None]
    """
    return ecdf.goal_f


def goal_to_str(goal_f: Union[int, float, None]) -> str:
    """
    Transform a goal to a string.

    :param Union[int, float, None] goal_f: the goal value
    :return: the string representation
    :rtype: str
    """
    return "undefined" if goal_f is None else \
        f"goal: \u2264{lg.num_to_str(goal_f)}"
