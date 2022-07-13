"""Providing a process without explicit logging with a single space."""
from typing import Union, Final

from moptipy.api._process_base import _ProcessBase, _TIME_IN_NS


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
