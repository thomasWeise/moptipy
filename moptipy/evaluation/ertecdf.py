"""Approximate the ECDF over the ERT to reach certain goals."""

from dataclasses import dataclass
from typing import List, Union

from moptipy.evaluation.ecdf import Ecdf
from moptipy.evaluation.ert import compute_single_ert
from moptipy.evaluation.plot_ert_impl import ert_y_axis_label
from moptipy.evaluation.progress import Progress


@dataclass(frozen=True, init=False, order=True)
class ErtEcdf(Ecdf):
    """The ERT-ECDF."""

    def time_label(self) -> str:
        """
        The method used to get the time axis label.

        :return: the time key
        :rtype: str
        """
        return ert_y_axis_label(self.time_unit)

    def _time_key(self) -> str:
        """
        The internal method used to get the time key.

        :return: the time key
        :rtype: str
        """
        return f"ert[{super()._time_key()}]"

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
        return [compute_single_ert(source, goal)]

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
        del n
        return n_insts
