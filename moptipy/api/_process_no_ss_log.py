"""A process with logging, where search and solution space are the same."""
from math import inf, isnan
from time import monotonic_ns
from typing import Optional, Union, List, Tuple

from moptipy.api._process_no_ss import _ProcessNoSS
from moptipy.api.algorithm import Algorithm
from moptipy.api.objective import Objective
from moptipy.api.space import Space
from moptipy.utils import logging
from moptipy.utils.logger import Logger


class _ProcessNoSSLog(_ProcessNoSS):
    """An process with logging, with the same search and solution space."""

    def __init__(self,
                 solution_space: Space,
                 objective: Objective,
                 algorithm: Algorithm,
                 log_file: str = None,
                 rand_seed: Optional[int] = None,
                 max_fes: Optional[int] = None,
                 max_time_millis: Optional[int] = None,
                 goal_f: Union[int, float, None] = None,
                 log_all_fes: bool = False) -> None:

        super().__init__(solution_space=solution_space,
                         objective=objective,
                         algorithm=algorithm,
                         log_file=log_file,
                         rand_seed=rand_seed,
                         max_fes=max_fes,
                         max_time_millis=max_time_millis,
                         goal_f=goal_f)
        if not isinstance(log_all_fes, bool):
            raise TypeError("log_all must be boolean, but is '"
                            + str(log_all_fes) + "'.")
        self.__log_all = log_all_fes
        self.__log: List[Tuple[int, int, Union[int, float]]] = list()

    def evaluate(self, x) -> Union[float, int]:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError('The process has been terminated and '
                                 'the algorithm knows it.')
            return inf

        result = self._objective.evaluate(x)
        if isnan(result):
            raise ValueError("NaN invalid as objective value.")

        self._current_fes += 1

        do_term = self._current_fes >= self._end_fes
        do_log = self.__log_all

        if (self._current_fes <= 1) or (result < self._current_best_f):
            self._last_improvement_fe = self._current_fes
            self._current_best_f = result
            needs_time_millis = False
            self._current_time_millis = int((monotonic_ns() + 999_999)
                                            // 1_000_000)
            self._last_improvement_time_millis = self._current_time_millis
            if self._current_time_millis >= self._end_time_millis:
                do_term = True
            self._solution_space.copy(x, self._current_best_y)
            self._has_current_best = True
            do_log = True
            if result <= self._end_f:
                do_term = True
        else:
            needs_time_millis = True

        if do_log and (not (self.__log is None)):
            if needs_time_millis:
                self._current_time_millis = int((monotonic_ns() + 999_999)
                                                // 1_000_000)
                if self._current_time_millis >= self._end_time_millis:
                    do_term = True
            self.__log.append((self._current_fes,
                               self._current_time_millis
                               - self._start_time_millis,
                               result))

        if do_term:
            self.terminate()

        return result

    def _write_log(self, logger: Logger) -> None:
        if len(self.__log) > 0:
            with logger.csv(logging.SECTION_PROGRESS,
                            [logging.PROGRESS_FES,
                             logging.PROGRESS_TIME_MILLIS,
                             logging.PROGRESS_CURRENT_F]) as csv:
                for row in self.__log:
                    csv.row(row)
        self.__log = None
        super()._write_log(logger)

    def get_name(self) -> str:
        return "LoggingProcessWithoutSearchSpace"
