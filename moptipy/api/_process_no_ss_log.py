"""A process with logging, where search and solution space are the same."""
from math import inf, isnan
from time import monotonic_ns
from typing import Optional, Union, List, Tuple, Final

from moptipy.api import logging
from moptipy.api._process_no_ss import _ProcessNoSS
from moptipy.api.algorithm import Algorithm
from moptipy.api.objective import Objective
from moptipy.api.space import Space
from moptipy.utils.logger import Logger
from moptipy.utils.path import Path


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

        :param Space solution_space: the search- and solution space.
        :param Objective objective: the objective function
        :param Algorithm algorithm: the optimization algorithm
        :param Path log_file: the optional log file
        :param Optional[int] rand_seed: the optional random seed
        :param Optional[int] max_fes: the maximum permitted function
            evaluations
        :param Optional[int] max_time_millis: the maximum runtime in
            milliseconds
        :param Union[int, float, None] goal_f: the goal objective
            value: if it is reached, the process is terminated
        :param bool log_all_fes: should we log all FEs?
        """
        super().__init__(solution_space=solution_space,
                         objective=objective,
                         algorithm=algorithm,
                         log_file=log_file,
                         rand_seed=rand_seed,
                         max_fes=max_fes,
                         max_time_millis=max_time_millis,
                         goal_f=goal_f)
        if not log_file:
            raise ValueError("Log file cannot be None in this class.")
        if not isinstance(log_all_fes, bool):
            raise TypeError(
                f"log_all must be bool, but is {type(log_all_fes)}.")

        #: `True` if all FEs are logged, `False` to only log improvements.
        self.__log_all: Final[bool] = log_all_fes
        #: The in-memory log
        self.__log: List[Tuple[int, int, Union[int, float]]] = []

    def evaluate(self, x) -> Union[float, int]:
        """
        Evaluate a candidate solution.

        This method internally forwards to :meth:`Objective.evaluate` of
        :attr:`_objective` and keeps track of the best-so-far solution.
        It also performs the logging of the progress.

        :param x: the candidate solution
        :return: the objective value
        :rtype: Union[float, int]
        """
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError('The process has been terminated and '
                                 'the algorithm knows it.')
            return inf

        result: Union[int, float] = self._objective.evaluate(x)
        if isnan(result):
            raise ValueError(
                f"NaN invalid as objective value, but got {result}.")

        self._current_fes += 1

        do_term: bool = self._current_fes >= self._end_fes
        do_log: bool = self.__log_all

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
