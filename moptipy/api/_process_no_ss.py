"""Providing a process without explicit logging with a single space."""
from typing import Union, Final, List, cast

from moptipy.api._process_base import _ProcessBase, _TIME_IN_NS, _ns_to_ms
from moptipy.api.logging import SECTION_PROGRESS, PROGRESS_FES, \
    PROGRESS_TIME_MILLIS, PROGRESS_CURRENT_F
from moptipy.utils.logger import Logger


class _ProcessNoSS(_ProcessBase):
    """
    An internal class process implementation.

    This class implements a stand-alone process without explicit logging where
    the search and solution space are the same.
    """

    def evaluate(self, x) -> Union[float, int]:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError('The process has been terminated and '
                                 'the algorithm knows it.')
            return self._current_best_f

        result: Final[Union[int, float]] = self._f(x)
        self._current_fes = current_fes = self._current_fes + 1
        do_term: bool = current_fes >= self._end_fes

        if result < self._current_best_f:
            self._last_improvement_fe = current_fes
            self._current_best_f = result
            self._current_time_nanos = ctn = _TIME_IN_NS()
            self._last_improvement_time_nanos = ctn
            do_term = do_term or (result <= self._end_f)
            self._copy_y(self._current_best_y, x)

        if do_term:
            self.terminate()

        return result

    def register(self, x, f: Union[int, float]) -> None:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError('The process has been terminated and '
                                 'the algorithm knows it.')
            return

        self._current_fes = current_fes = self._current_fes + 1
        do_term: bool = current_fes >= self._end_fes

        if f < self._current_best_f:
            self._last_improvement_fe = current_fes
            self._current_best_f = f
            self._current_time_nanos = ctn = _TIME_IN_NS()
            self._last_improvement_time_nanos = ctn
            do_term = do_term or (f <= self._end_f)
            self._copy_y(self._current_best_y, x)

        if do_term:
            self.terminate()

    def __str__(self) -> str:
        return "ProcessWithoutSearchSpace"


def _write_log(log: List[List[Union[int, float]]],
               start_time: int,
               logger: Logger) -> None:
    """
    Write the log to a logger.

    :param log: the log
    :param start_time: the start time
    :param logger: the logger
    """
    if len(log) > 0:
        with logger.csv(SECTION_PROGRESS,
                        [PROGRESS_FES,
                         PROGRESS_TIME_MILLIS,
                         PROGRESS_CURRENT_F]) as csv:
            for row in log:
                csv.row([row[0], _ns_to_ms(cast(int, row[1]) - start_time),
                         row[2]])
