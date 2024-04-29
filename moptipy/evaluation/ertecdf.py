"""
Approximate the ECDF over the ERT to reach certain goals.

The empirical cumulative distribution function (ECDF, see
:mod:`~moptipy.evaluation.ecdf`) is a function that shows the fraction of runs
that were successful in attaining a certain goal objective value over the
time. The (empirically estimated) Expected Running Time (ERT, see
:mod:`~moptipy.evaluation.ert`) is a function that tries to give an estimate
how long a given algorithm setup will need (y-axis) to achieve given solution
qualities (x-axis). It uses a set of runs of the algorithm on the problem to
make this estimate under the assumption of independent restarts.

Now in the ERT-ECDF we combine both concepts to join several different
optimization problems or problem instances into one plot. The goal becomes
"solving the problem". For each problem instance, we compute the ERT, i.e.,
estimate how long a given algorithm will need to reach the goal. This becomes
the time axis. Over this time axis, the ERT-ECDF displays the fraction of
instances that were solved.

1. Thomas Weise, Zhize Wu, Xinlu Li, and Yan Chen. Frequency Fitness
   Assignment: Making Optimization Algorithms Invariant under Bijective
   Transformations of the Objective Function Value. *IEEE Transactions on
   Evolutionary Computation* 25(2):307-319. April 2021. Preprint available at
   arXiv:2001.01416v5 [cs.NE] 15 Oct 2020. http://arxiv.org/abs/2001.01416.
   doi: https://doi.org/10.1109/TEVC.2020.3032090
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from moptipy.evaluation.ecdf import Ecdf
from moptipy.evaluation.ert import compute_single_ert
from moptipy.evaluation.progress import Progress


@dataclass(frozen=True, init=False, order=False, eq=False)
class ErtEcdf(Ecdf):
    """The ERT-ECDF."""

    def time_label(self) -> str:
        """
        Get the time axis label.

        :return: the time key
        """
        return f"ERT\u2009[{self.time_unit}]"

    def _time_key(self) -> str:
        """
        Get the time key.

        :return: the time key
        """
        return f"ert[{super()._time_key()}]"

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
        return [compute_single_ert(source, goal)]

    # noinspection PyUnusedLocal
    @staticmethod
    def _get_div(n: int, n_insts: int) -> int:
        """
        Get the divisor.

        :param n: the number of runs
        :param n_insts: the number of instances
        :return: the divisor
        """
        del n
        return n_insts


def create(source: Iterable[Progress],
           goal_f: int | float | Callable[[str], int | float] | None = None,
           use_default_goal_f: bool = True) -> Ecdf:
    """
    Create one single Ert-Ecdf record from an iterable of Progress records.

    :param source: the set of progress instances
    :param goal_f: the goal objective value
    :param use_default_goal_f: should we use the default lower bounds as
        goals?
    :return: the Ert-Ecdf record
    """
    return ErtEcdf._create(source, goal_f, use_default_goal_f)


def from_progresses(
        source: Iterable[Progress], consumer: Callable[[Ecdf], Any],
        f_goal: int | float | Callable[[str], int | float]
                    | Iterable[int | float | Callable] | None = None,
        join_all_algorithms: bool = False,
        join_all_objectives: bool = False,
        join_all_encodings: bool = False) -> None:
    """
    Compute one or multiple Ert-ECDFs from a stream of end results.

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
    return ErtEcdf._from_progresses(
        source, consumer, f_goal, join_all_algorithms,
        join_all_objectives, join_all_encodings)
