"""
Approximate the :class:`~moptipy.evaluation.ecdf.Ecdf` to reach certain goals.

The empirical cumulative distribution function (ECDF) for short illustrates
the fraction of runs that have reached a certain goal over time. Let's say
that you have performed 10 runs of a certain algorithm on a certain problem.
As goal quality, you could define the globally optimal solution quality.
For any point in time, the ECDF then shows how many of these runs have solved
the problem to this goal, to optimality.
Let's say the first run solves the problem after 100 FEs.
Then the ECDF is 0 until 99 FEs and at 100 FEs, it becomes 1/10.
The second-fastest run solves the problem after 200 FEs.
The ECDF thus stays 0.1 until 199 FEs and at 200 FEs, it jumps to 0.2.
And so on.
This means that the value of the ECDF is always between 0 and 1.

1. Nikolaus Hansen, Anne Auger, Steffen Finck, Raymond Ros. *Real-Parameter
   Black-Box Optimization Benchmarking 2010: Experimental Setup.*
   Research Report RR-7215, INRIA. 2010. inria-00462481.
   https://hal.inria.fr/inria-00462481/document/
2. Dave Andrew Douglas Tompkins and Holger H. Hoos. UBCSAT: An Implementation
   and Experimentation Environment for SLS Algorithms for SAT and MAX-SAT. In
   *Revised Selected Papers from the Seventh International Conference on
   Theory and Applications of Satisfiability Testing (SAT'04),* May 10-13,
   2004, Vancouver, BC, Canada, pages 306-320. Lecture Notes in Computer
   Science (LNCS), volume 3542. Berlin, Germany: Springer-Verlag GmbH.
   ISBN: 3-540-27829-X. doi: https://doi.org/10.1007/11527695_24.
3. Holger H. Hoos and Thomas Stützle. Evaluating Las Vegas Algorithms -
   Pitfalls and Remedies. In Gregory F. Cooper and Serafín Moral, editors,
   *Proceedings of the 14th Conference on Uncertainty in Artificial
   Intelligence (UAI'98)*, July 24-26, 1998, Madison, WI, USA, pages 238-245.
   San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.
   ISBN: 1-55860-555-X.
"""

from dataclasses import dataclass
from math import inf, isfinite
from typing import Any, Callable, Final, Iterable

import numpy as np
from pycommons.io.console import logger
from pycommons.io.csv import COMMENT_START, CSV_SEPARATOR
from pycommons.io.path import Path
from pycommons.strings.string_conv import num_to_str
from pycommons.types import check_int_range, type_error

from moptipy.api.logging import (
    KEY_ALGORITHM,
    KEY_GOAL_F,
)
from moptipy.evaluation._utils import _get_goal_reach_index
from moptipy.evaluation.base import (
    F_NAME_NORMALIZED,
    F_NAME_SCALED,
    KEY_ENCODING,
    KEY_N,
    KEY_OBJECTIVE_FUNCTION,
    MultiRun2DData,
)
from moptipy.evaluation.progress import Progress
from moptipy.utils.lang import Lang
from moptipy.utils.logger import (
    KEY_VALUE_SEPARATOR,
)
from moptipy.utils.nputils import is_all_finite

#: The number of instances.
KEY_N_INSTS: Final[str] = f"{KEY_N}Insts"
#: The objective dimension name.
KEY_F_NAME: Final[str] = "fName"


@dataclass(frozen=True, init=False, order=False, eq=False)
class Ecdf(MultiRun2DData):
    """The ECDF data."""

    #: The ECDF data function
    ecdf: np.ndarray
    #: The number of instances over which the ERT-ECDF is computed.
    n_insts: int
    #: The goal value, or None if different goals were used for different
    #: instances
    goal_f: int | float | None

    def __init__(self,
                 algorithm: str | None,
                 objective: str | None,
                 encoding: str | None,
                 n: int,
                 n_insts: int,
                 time_unit: str,
                 f_name: str,
                 goal_f: int | float | None,
                 ecdf: np.ndarray):
        """
        Create the ECDF function.

        :param algorithm: the algorithm name, if all runs are with the same
            algorithm
        :param objective: the objective name, if all runs are on the same
            objective function, `None` otherwise
        :param encoding: the encoding name, if all runs are on the same
            encoding and an encoding was actually used, `None` otherwise
        :param n: the total number of runs
        :param n_insts: the total number of instances
        :param time_unit: the time unit
        :param f_name: the objective dimension name
        :param goal_f: the goal value, or `None` if different goals were used
            for different instances
        :param numpy.ndarray ecdf: the ert-ecdf matrix
        """
        super().__init__(algorithm, None, objective, encoding, n,
                         time_unit, f_name)
        object.__setattr__(
            self, "n_insts", check_int_range(n_insts, "n_insts", 1, self.n))

        if goal_f is not None:
            if not isinstance(goal_f, int | float):
                raise type_error(goal_f, "goal_f", (int, float))
            if not isfinite(goal_f):
                raise ValueError(f"Invalid goal_f {goal_f}.")
        object.__setattr__(self, "goal_f", goal_f)

        if not isinstance(ecdf, np.ndarray):
            raise type_error(ecdf, "ecdf", np.ndarray)
        ecdf.flags.writeable = False

        time: Final[np.ndarray] = ecdf[:, 0]
        ll = time.size
        if ll < 2:
            raise ValueError("Must have at least two points in "
                             f"ecdf curve , but encountered {ll}.")
        if not is_all_finite(time[:-1]):
            raise ValueError("Non-last Ert-based time-values must be finite, "
                             f"but encountered {time}.")
        if np.any(time[1:] <= time[:-1]):
            raise ValueError("Time data must be strictly increasing,"
                             f"but encountered {time}.")

        prb: Final[np.ndarray] = ecdf[:, 1]
        if not is_all_finite(prb):
            raise ValueError(
                f"All ECDF values must be finite, but encountered {prb}.")
        if (ll > 2) and np.any(prb[1:-1] <= prb[:-2]):
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
        """
        return Lang.translate(self.time_unit)

    def _time_key(self) -> str:
        """
        Get the time key.

        :return: the time key
        """
        return self.time_unit

    def _tuple(self) -> tuple[Any, ...]:
        """
        Get the tuple representation of this object used in comparisons.

        :return: the comparison-relevant data of this object in a tuple
        """
        return (self.__class__.__name__,
                "" if self.algorithm is None else self.algorithm,
                "" if self.instance is None else self.instance,
                "" if self.objective is None else self.objective,
                "" if self.encoding is None else self.encoding,
                self.n, -1, self.time_unit, self.f_name,
                inf if self.goal_f is None else self.goal_f,
                self.n_insts)

    def to_csv(self, file: str,
               put_header: bool = True) -> Path:
        """
        Store a :class:`Ecdf` record in a CSV file.

        :param file: the file to generate
        :param put_header: should we put a header with meta-data?
        :return: the fully resolved file name
        """
        path: Final[Path] = Path(file)
        logger(f"Writing ECDF to CSV file {path!r}.")
        path.ensure_parent_dir_exists()

        with path.open_for_write() as out:
            sep: Final[str] = CSV_SEPARATOR
            write: Final[Callable[[str], int]] = out.write
            if put_header:
                kv: Final[str] = KEY_VALUE_SEPARATOR
                cmt: Final[str] = COMMENT_START
                if self.algorithm is not None:
                    write(f"{cmt} {KEY_ALGORITHM}{kv}{self.algorithm}\n")
                write(f"{cmt} {KEY_N}{kv}{self.n}\n")
                write(f"{cmt} {KEY_N_INSTS}{kv}{self.n_insts}\n")
                write(f"{cmt} {KEY_F_NAME}{kv}{self.f_name}\n")
                if self.goal_f is not None:
                    write(f"{cmt} {KEY_GOAL_F}{kv}{self.goal_f}\n")
                if self.objective is not None:
                    write(f"{cmt} {KEY_OBJECTIVE_FUNCTION}"
                          f"{kv}{self.objective}\n")
                if self.encoding is not None:
                    write(f"{cmt} {KEY_ENCODING}{kv}{self.encoding}\n")

            write(f"{self._time_key()}{sep}ecdf\n")
            for v in self.ecdf:
                out.write(
                    f"{num_to_str(v[0])}{sep}{num_to_str(v[1])}\n")

        logger(f"Done writing ECDF to CSV file {path!r}.")

        path.enforce_file()
        return path

    @staticmethod
    def _compute_times(source: list[Progress],
                       goal: int | float) -> list[float]:
        """
        Compute the times for the given goals.

        Warning: `source` must only contain progress objects that contain
        monotonously improving points. It must not contain runs that may get
        worse over time.

        :param source: the source array
        :param goal: the goal value
        :return: a list of times
        """
        ret = []
        for pr in source:
            idx = _get_goal_reach_index(pr.f, goal)
            if idx < 0:
                continue
            ret.append(pr.time[idx])
        return ret

    # noinspection PyUnusedLocal
    @staticmethod
    def _get_div(n: int, n_insts: int) -> int:
        """
        Get the divisor.

        :param n: the number of runs
        :param n_insts: the number of instances
        :return: the divisor
        """
        del n_insts
        return n

    @classmethod
    def create(cls: type["Ecdf"],
               source: Iterable[Progress],
               goal_f: int | float | Callable | None = None,
               use_default_goal_f: bool = True) -> "Ecdf":
        """
        Create one single Ecdf record from an iterable of Progress records.

        :param source: the set of progress instances
        :param goal_f: the goal objective value
        :param use_default_goal_f: should we use the default lower bounds as
            goals?
        :return: the Ecdf record
        :rtype: Ecdf
        """
        if not isinstance(source, Iterable):
            raise type_error(source, "source", Iterable)

        algorithm: str | None = None
        objective: str | None = None
        encoding: str | None = None
        time_unit: str | None = None
        f_name: str | None = None
        inst_runs: dict[str, list[Progress]] = {}
        n: int = 0

        for progress in source:
            if not isinstance(progress, Progress):
                raise type_error(progress, "progress", Progress)
            if n <= 0:
                algorithm = progress.algorithm
                time_unit = progress.time_unit
                objective = progress.objective
                encoding = progress.encoding
                f_name = progress.f_name
            else:
                if algorithm != progress.algorithm:
                    algorithm = None
                if objective != progress.objective:
                    objective = None
                if encoding != progress.encoding:
                    encoding = None
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

        times: list[float] = []
        goal: int | float | None
        same_goal_f: int | float | None = None
        first: bool = True
        for instance, pl in inst_runs.items():
            goal = None
            if isinstance(goal_f, int | float):
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
            if not isinstance(goal, int | float):
                raise type_error(goal, "goal", (int, float))
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
            return cls(algorithm, objective, encoding, n, n_insts, time_unit,
                       f_name, same_goal_f, np.array([[0, 0], [inf, 0]]))

        times.sort()
        time: list[float] = [0]
        ecdf: list[float] = [0]
        success: int = 0
        div: Final[int] = cls._get_div(n, n_insts)
        ll: int = 0
        for t in times:
            success += 1  # noqa: SIM113
            if t > time[ll]:
                time.append(t)
                ecdf.append(success / div)
                ll += 1
            else:
                ecdf[ll] = success / div

        time.append(inf)
        ecdf.append(ecdf[ll])

        return cls(algorithm, objective, encoding, n, n_insts,
                   time_unit, f_name, same_goal_f,
                   np.column_stack((np.array(time), np.array(ecdf))))

    @classmethod
    def from_progresses(
            cls: type["Ecdf"],
            source: Iterable[Progress], consumer: Callable[["Ecdf"], Any],
            f_goal: int | float | Callable
                        | Iterable[int | float | Callable] | None = None,
            join_all_algorithms: bool = False,
            join_all_objectives: bool = False,
            join_all_encodings: bool = False) -> None:
        """
        Compute one or multiple ECDFs from a stream of end results.

        :param source: the set of progress instances
        :param f_goal: one or multiple goal values
        :param consumer: the destination to which the new records will be
            passed, can be the `append` method of a :class:`list`
        :param join_all_algorithms: should the Ecdf be aggregated over all
            algorithms
        :param join_all_objectives: should the Ecdf be aggregated over all
            objective functions
        :param join_all_encodings: should the Ecdf be aggregated over all
            encodings
        """
        if not isinstance(source, Iterable):
            raise type_error(source, "source", Iterable)
        if not callable(consumer):
            raise type_error(consumer, "consumer", call=True)
        if not isinstance(join_all_algorithms, bool):
            raise type_error(join_all_algorithms,
                             "join_all_algorithms", bool)
        if not isinstance(join_all_objectives, bool):
            raise type_error(join_all_objectives,
                             "join_all_objectives", bool)
        if not isinstance(join_all_encodings, bool):
            raise type_error(join_all_encodings,
                             "join_all_encodings", bool)
        if not isinstance(f_goal, Iterable):
            f_goal = [f_goal]

        sorter: dict[tuple[str, str, str], list[Progress]] = {}
        for er in source:
            if not isinstance(er, Progress):
                raise type_error(er, "progress", Progress)
            key = ("" if join_all_algorithms else er.algorithm,
                   "" if join_all_objectives else er.objective,
                   "" if join_all_encodings else (
                       "" if er.encoding is None else er.encoding))
            if key in sorter:
                lst = sorter[key]
            else:
                lst = []
                sorter[key] = lst
            lst.append(er)

        if len(sorter) <= 0:
            raise ValueError("source must not be empty")

        keyz = sorted(sorter.keys())
        for goal in f_goal:
            use_default_goal = goal is None
            for key in keyz:
                consumer(cls.create(sorter[key], goal, use_default_goal))


def get_goal(ecdf: Ecdf) -> int | float | None:
    """
    Get the goal value from the given ecdf instance.

    :param Ecdf ecdf: the ecdf instance
    :return: the goal value
    """
    return ecdf.goal_f


def goal_to_str(goal_f: int | float | None) -> str:
    """
    Transform a goal to a string.

    :param goal_f: the goal value
    :return: the string representation
    """
    return "undefined" if goal_f is None else \
        f"goal: \u2264{num_to_str(goal_f)}"
