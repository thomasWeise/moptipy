"""An implementation of processes with different search and solution spaces."""
from math import inf, isnan
from time import monotonic_ns
from typing import Optional, Union, Final

from moptipy.api import logging
from moptipy.api._process_no_ss import _ProcessNoSS
from moptipy.api.algorithm import Algorithm
from moptipy.api.encoding import Encoding, check_encoding
from moptipy.api.objective import Objective
from moptipy.api.space import Space, check_space
from moptipy.utils.logger import KeyValueSection, Logger
from moptipy.utils.path import Path


class _ProcessSS(_ProcessNoSS):
    """A class implementing a process with search and solution space."""

    def __init__(self,
                 solution_space: Space,
                 objective: Objective,
                 algorithm: Algorithm,
                 log_file: Optional[Path] = None,
                 search_space: Space = None,
                 encoding: Encoding = None,
                 rand_seed: Optional[int] = None,
                 max_fes: Optional[int] = None,
                 max_time_millis: Optional[int] = None,
                 goal_f: Union[int, float, None] = None) -> None:
        """
        Perform the internal initialization. Do not call directly.

        :param Space solution_space: the solution space.
        :param Objective objective: the objective function
        :param Algorithm algorithm: the optimization algorithm
        :param Space search_space: the search space.
        :param Encoding encoding: the encoding
        :param Optional[Path] log_file: the optional log file
        :param Optional[int] rand_seed: the optional random seed
        :param Optional[int] max_fes: the maximum permitted function
            evaluations
        :param Optional[int] max_time_millis: the maximum runtime in
            milliseconds
        :param Union[int, float, None] goal_f: the goal objective
            value: if it is reached, the process is terminated
        """
        super().__init__(solution_space=solution_space,
                         objective=objective,
                         algorithm=algorithm,
                         log_file=log_file,
                         rand_seed=rand_seed,
                         max_fes=max_fes,
                         max_time_millis=max_time_millis,
                         goal_f=goal_f)

        #: The search space.
        self._search_space: Final[Space] = check_space(search_space)
        #: The encoding.
        self._encoding: Final[Encoding] = check_encoding(encoding)
        #: The holder for the currently de-coded solution.
        self._current_y: Final = self._solution_space.create()
        #: The current best point in the search space.
        self._current_best_x: Final = self._search_space.create()

    def evaluate(self, x) -> Union[float, int]:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError('The process has been terminated and the '
                                 'algorithm knows it.')
            return inf

        self._encoding.map(x, self._current_y)
        result: Union[int, float] = self._objective.evaluate(self._current_y)
        if isnan(result):
            raise ValueError(
                f"NaN invalid as objective value, but got {result}.")
        self._current_fes += 1

        do_term: bool = self._current_fes >= self._end_fes

        if (self._current_fes <= 1) or (result < self._current_best_f):
            # noinspection PyAttributeOutsideInit
            self._last_improvement_fe = self._current_fes
            self._current_best_f = result
            self._search_space.copy(x, self._current_best_x)
            self._solution_space.copy(self._current_y, self._current_best_y)
            self._current_time_millis = int((monotonic_ns() + 999_999)
                                            // 1_000_000)
            self._last_improvement_time_millis = self._current_time_millis
            if self._current_time_millis >= self._end_time_millis:
                do_term = True

            # noinspection PyAttributeOutsideInit
            self._has_current_best = True
            if result <= self._end_f:
                do_term = True

        if do_term:
            self.terminate()

        return result

    def has_current_best(self) -> bool:
        return self._has_current_best

    def get_current_best_f(self) -> float:
        if self._has_current_best:
            return self._current_best_f
        raise ValueError('No current best available.')

    def get_copy_of_current_best_x(self, x) -> None:
        if self._has_current_best:
            return self._search_space.copy(self._current_best_x, x)
        raise ValueError('No current best available.')

    def create(self):
        return self._search_space.create()

    def copy(self, source, dest):
        self._search_space.copy(source, dest)

    def to_str(self, x) -> str:
        return self._search_space.to_str(x)

    def from_str(self, text: str):
        return self._search_space.from_str(text)

    def is_equal(self, x1, x2) -> bool:
        return self._search_space.is_equal(x1, x2)

    def validate(self, x) -> None:
        self._search_space.validate(x)

    def get_copy_of_current_best_y(self, y):
        return self._solution_space.copy(self._current_best_y, y)

    def scale(self) -> int:
        return self._search_space.scale()

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        super().log_parameters_to(logger)
        with logger.scope(logging.SCOPE_SEARCH_SPACE) as sc:
            self._search_space.log_parameters_to(sc)
        with logger.scope(logging.SCOPE_ENCODING) as sc:
            self._encoding.log_parameters_to(sc)

    def _write_log(self, logger: Logger) -> None:
        # noinspection PyProtectedMember
        super()._write_log(logger)

        if self._has_current_best:
            with logger.text(logging.SECTION_RESULT_X) as txt:
                txt.write(self._search_space.to_str(self._current_best_x))

    def _perform_termination(self) -> None:
        # noinspection PyProtectedMember
        super()._perform_termination()
        self._search_space.validate(self._current_best_x)

    def get_name(self) -> str:
        return "ProcessWithSearchSpace"
