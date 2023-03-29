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

from moptipy.evaluation.ecdf import Ecdf
from moptipy.evaluation.ert import compute_single_ert
from moptipy.evaluation.progress import Progress


@dataclass(frozen=True, init=False, order=True)
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
