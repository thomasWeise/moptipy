"""An implementation of processes with different search and solution spaces."""
from typing import Callable, Final, cast

from pycommons.io.path import Path

from moptipy.api._process_base import _TIME_IN_NS
from moptipy.api._process_no_ss import _ProcessNoSS
from moptipy.api.algorithm import Algorithm
from moptipy.api.encoding import Encoding, check_encoding
from moptipy.api.improvement_logger import ImprovementLogger
from moptipy.api.logging import (
    SCOPE_ENCODING,
    SCOPE_SEARCH_SPACE,
    SECTION_RESULT_X,
)
from moptipy.api.objective import Objective
from moptipy.api.space import Space, check_space
from moptipy.utils.logger import KeyValueLogSection, Logger


class _ProcessSS(_ProcessNoSS):
    """A class implementing a process with search and solution space."""

    def __init__(self,
                 solution_space: Space,
                 objective: Objective,
                 algorithm: Algorithm,
                 log_file: Path | None = None,
                 search_space: Space | None = None,
                 encoding: Encoding | None = None,
                 rand_seed: int | None = None,
                 max_fes: int | None = None,
                 max_time_millis: int | None = None,
                 goal_f: int | float | None = None,
                 improvement_logger: ImprovementLogger | None = None) -> None:
        """
        Perform the internal initialization. Do not call directly.

        :param solution_space: the solution space.
        :param objective: the objective function
        :param algorithm: the optimization algorithm
        :param search_space: the search space.
        :param encoding: the encoding
        :param log_file: the optional log file
        :param rand_seed: the optional random seed
        :param max_fes: the maximum permitted function evaluations
        :param max_time_millis: the maximum runtime in milliseconds
        :param goal_f: the goal objective value. if it is reached, the
            process is terminated
        :param improvement_logger: an improvement logger, whose
            :meth:`~ImprovementLogger.log_improvement` method will be invoked
            whenever the process has registered an improvement
        """
        super().__init__(solution_space=solution_space,
                         objective=objective,
                         algorithm=algorithm,
                         log_file=log_file,
                         rand_seed=rand_seed,
                         max_fes=max_fes,
                         max_time_millis=max_time_millis,
                         goal_f=goal_f,
                         improvement_logger=improvement_logger)

        #: The search space.
        self._search_space: Final[Space] = check_space(search_space)
        #: The encoding.
        self._encoding: Final[Encoding] = check_encoding(encoding)
        #: the internal encoder
        self._g: Final[Callable] = encoding.decode
        #: The holder for the currently de-coded solution.
        self._current_y = solution_space.create()
        #: The current best point in the search space.
        self._current_best_x: Final = search_space.create()
        # wrappers
        self.create = search_space.create  # type: ignore
        self.copy = search_space.copy  # type: ignore
        self.to_str = search_space.to_str  # type: ignore
        self.is_equal = search_space.is_equal  # type: ignore
        self.from_str = search_space.from_str  # type: ignore
        self.n_points = search_space.n_points  # type: ignore
        self.validate = search_space.validate  # type: ignore

    def evaluate(self, x) -> float | int:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError("The process has been terminated and the "
                                 "algorithm knows it.")
            return self._current_best_f

        current_y: Final = self._current_y
        self._g(x, current_y)
        result: Final[int | float] = self._f(current_y)
        self._current_fes = current_fes = self._current_fes + 1
        do_term: bool = current_fes >= self._end_fes

        if result < self._current_best_f:
            self._last_improvement_fe = current_fes
            self._current_best_f = result
            self.copy(self._current_best_x, x)
            self._current_y = self._current_best_y
            self._current_best_y = current_y
            self._current_time_nanos = ctn = _TIME_IN_NS()
            self._last_improvement_time_nanos = ctn
            do_term = do_term or (result <= self._end_f)
            if self._log_improvement:
                self._log_improvement(
                    cast("Callable[[Logger], None]",
                         lambda lg, _x=x, _y=current_y, _f=result:
                         self._write_improvement(lg, _x, _y, _f)))

        if do_term:
            self.terminate()

        return result

    def register(self, x, f: int | float) -> None:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError("The process has been terminated and the "
                                 "algorithm knows it.")
            return

        self._current_fes = current_fes = self._current_fes + 1
        do_term: bool = current_fes >= self._end_fes

        if f < self._current_best_f:
            self._last_improvement_fe = current_fes
            self._current_best_f = f
            self.copy(self._current_best_x, x)
            current_y: Final = self._current_y
            self._g(x, current_y)
            self._current_y = self._current_best_y
            self._current_best_y = current_y
            self._current_time_nanos = ctn = _TIME_IN_NS()
            self._last_improvement_time_nanos = ctn
            do_term = do_term or (f <= self._end_f)
            if self._log_improvement:
                self._log_improvement(
                    cast("Callable[[Logger], None]",
                         lambda lg, _x=x, _y=current_y, _f=f:
                         self._write_improvement(lg, _x, _y, _f)))

        if do_term:
            self.terminate()

    def get_copy_of_best_x(self, x) -> None:
        if self._current_fes > 0:
            return self.copy(x, self._current_best_x)
        raise ValueError("No current best x available.")

    def get_copy_of_best_y(self, y) -> None:
        if self._current_fes > 0:
            return self._copy_y(y, self._current_best_y)
        raise ValueError("No current best y available.")

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_SEARCH_SPACE) as sc:
            self._search_space.log_parameters_to(sc)
        with logger.scope(SCOPE_ENCODING) as sc:
            self._encoding.log_parameters_to(sc)

    def _write_result(self, logger: Logger) -> None:
        with logger.text(SECTION_RESULT_X) as txt:
            txt.write(self._search_space.to_str(self._current_best_x))
        super()._write_result(logger)

    def _validate_x(self) -> None:
        """Validate x, if it exists."""
        self._search_space.validate(self._current_best_x)

    def __str__(self) -> str:
        return "ProcessWithSearchSpace"

    def _write_improvement(
            self, logger: Logger, x, y, f: int | float) -> None:
        """
        Write an improvement to the logger.

        :param logger: the logger
        :param x: the point in the search space
        :param y: the point in the solution space
        :param f: the objective value
        """
        super()._write_improvement(logger, self._search_space.to_str(x), y, f)
