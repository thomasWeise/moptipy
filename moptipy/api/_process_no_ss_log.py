"""A process with logging, where search and solution space are the same."""
from typing import Optional, Union, List, Final, cast

from moptipy.api import logging
from moptipy.api._process_base import _ns_to_ms, _TIME_IN_NS
from moptipy.api._process_no_ss import _ProcessNoSS
from moptipy.api.algorithm import Algorithm
from moptipy.api.objective import Objective
from moptipy.api.space import Space
from moptipy.utils.logger import Logger
from moptipy.utils.path import Path
from moptipy.utils.types import type_error


class _ProcessNoSSLog(_ProcessNoSS):
    """An process with logging, with the same search and solution space."""

    def __init__(self,
                 solution_space: Space,
                 objective: Objective,
                 algorithm: Algorithm,
                 log_file: Path,
                 rand_seed: Optional[int] = None,
                 max_fes: Optional[int] = None,
                 max_time_millis: Optional[int] = None,
                 goal_f: Union[int, float, None] = None,
                 log_all_fes: bool = False) -> None:
        """
        Perform the internal initialization. Do not call directly.

        :param solution_space: the search- and solution space.
        :param objective: the objective function
        :param algorithm: the optimization algorithm
        :param log_file: the optional log file
        :param rand_seed: the optional random seed
        :param max_fes: the maximum permitted function evaluations
        :param max_time_millis: the maximum runtime in milliseconds
        :param goal_f: the goal objective value. if it is reached, the process
            is terminated
        :param log_all_fes: should we log all FEs?
        """
        super().__init__(solution_space=solution_space,
                         objective=objective,
                         algorithm=algorithm,
                         log_file=log_file,
                         rand_seed=rand_seed,
                         max_fes=max_fes,
                         max_time_millis=max_time_millis,
                         goal_f=goal_f)
        if not isinstance(log_file, str):
            raise type_error(log_file, "log_file", str)
        if not isinstance(log_all_fes, bool):
            raise type_error(log_all_fes, "log_all_fes", bool)

        #: `True` if all FEs are logged, `False` to only log improvements.
        self.__log_all: Final[bool] = log_all_fes
        #: The in-memory log
        self.__log: List[List[Union[int, float]]] = []
        #: the quick access to the log appending method
        self.__log_append = self.__log.append

    def evaluate(self, x) -> Union[float, int]:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError('The process has been terminated and '
                                 'the algorithm knows it.')
            return self._current_best_f

        result: Final[Union[int, float]] = self._f(x)
        self._current_fes = current_fes = self._current_fes + 1
        do_term: bool = current_fes >= self._end_fes
        do_log: bool = self.__log_all
        ctn: int = 0

        if result < self._current_best_f:
            self._last_improvement_fe = current_fes
            self._current_best_f = result
            self._current_time_nanos = ctn = _TIME_IN_NS()
            self._last_improvement_time_nanos = ctn
            do_term = do_term or (result <= self._end_f)
            self._copy_y(self._current_best_y, x)
            do_log = True

        if do_log:
            if ctn <= 0:
                self._current_time_nanos = ctn = _TIME_IN_NS()
            self.__log_append([current_fes, ctn, result])

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
        do_log: bool = self.__log_all
        ctn: int = 0

        if f < self._current_best_f:
            self._last_improvement_fe = current_fes
            self._current_best_f = f
            self._current_time_nanos = ctn = _TIME_IN_NS()
            self._last_improvement_time_nanos = ctn
            do_term = do_term or (f <= self._end_f)
            self._copy_y(self._current_best_y, x)
            do_log = True

        if do_log:
            if ctn <= 0:
                self._current_time_nanos = ctn = _TIME_IN_NS()
            self.__log_append([current_fes, ctn, f])

        if do_term:
            self.terminate()

    def _check_timing(self) -> None:
        super()._check_timing()
        last_time: int = -1
        last_fe: int = -1
        start_time: Final[int] = self._start_time_nanos
        current_time: Final[int] = self._current_time_nanos
        for row in self.__log:
            fes: int = cast(int, row[0])
            time: int = cast(int, row[1])
            if fes < last_fe:
                raise ValueError(f"fe={fes} after fe={last_fe}?")
            if time < last_time:
                raise ValueError(
                    f"time={time} of fe={fes} is less than "
                    f"last_time={last_time} of last_fe={last_fe}")
            if time < start_time:
                raise ValueError(
                    f"time={time} of fe={fes} is less than "
                    f"start_time_nanos={start_time}")
            if time > current_time:
                raise ValueError(
                    f"time={time} of fe={fes} is greater than "
                    f"current_time_nanos={current_time}")
            last_time = time
            last_fe = fes

    def _write_log(self, logger: Logger) -> None:
        if len(self.__log) > 0:
            with logger.csv(logging.SECTION_PROGRESS,
                            [logging.PROGRESS_FES,
                             logging.PROGRESS_TIME_MILLIS,
                             logging.PROGRESS_CURRENT_F]) as csv:
                for row in self.__log:
                    csv.row([row[0], _ns_to_ms(cast(int, row[1])
                                               - self._start_time_nanos),
                             row[2]])
        del self.__log
        super()._write_log(logger)

    def __str__(self) -> str:
        return "LoggingProcessWithoutSearchSpace"
