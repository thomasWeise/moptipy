"""Providing a process without explicit logging with a single space."""
import os
from io import StringIO
from math import inf, isfinite
from traceback import print_tb
from typing import Optional, Union, Final, Callable, Dict

from numpy.random import Generator

from moptipy.api import logging
from moptipy.api._process_base import _ProcessBase, _ns_to_ms, _TIME_IN_NS
from moptipy.api.algorithm import Algorithm, check_algorithm
from moptipy.api.objective import Objective, check_objective
from moptipy.api.process import check_goal_f
from moptipy.api.space import Space, check_space
from moptipy.utils.logger import KeyValueLogSection, FileLogger
from moptipy.utils.logger import Logger
from moptipy.utils.nputils import rand_generator, rand_seed_generate, \
    rand_seed_check
from moptipy.utils.path import Path
from moptipy.utils.sys_info import log_sys_info
from moptipy.utils.types import type_name_of, type_error


def _error_1(logger: Logger, title: str, exception_type,
             exception_value, traceback):
    """
    Create a text section with error information as from a contextmanager.

    :param logger: the logger to write to
    :param title: the title of the section with error information to be
        created
    :param exception_type: the exception type
    :param exception_value: the exception value
    :param traceback: the traceback
    """
    if exception_type or exception_value or traceback:
        with logger.text(title=title) as ts:
            if exception_type:
                ts.write(logging.KEY_EXCEPTION_TYPE)
                ts.write(": ")
                if isinstance(exception_type, str):
                    if exception_type.startswith("<class '"):
                        exception_type = exception_type[8:-2]
                    ts.write(exception_type.strip())
                else:
                    ts.write(type_name_of(exception_type))
                ts.write(os.linesep)
            if exception_value:
                ts.write(logging.KEY_EXCEPTION_VALUE)
                ts.write(": ")
                ts.write(exception_value.strip())
                ts.write(os.linesep)
            if traceback:
                ts.write(logging.KEY_EXCEPTION_STACK_TRACE)
                ts.write(":")
                ts.write(os.linesep)
                sio = StringIO()
                print_tb(traceback, file=sio)
                for line in sio.getvalue().split("\n"):
                    ts.write(line.strip())
                    ts.write(os.linesep)


def _error_2(logger: Logger, title: str, exception: BaseException):
    """
    Log an exception.

    :param logger: the logger to write to
    :param title: the title of the section with error information to be
        created
    :param exception: the exception

    >>> from moptipy.utils.logger import InMemoryLogger
    >>> ime = InMemoryLogger()
    >>> def k():
    ...     1 / 0
    >>> try:
    ...     k()
    ... except BaseException as be:
    ...     _error_2(ime, "ERROR", be)
    >>> print(ime.get_log()[0])
    BEGIN_ERROR
    >>> print(ime.get_log()[1])
    exceptionType: ZeroDivisionError
    >>> print(ime.get_log()[2])
    exceptionValue: division by zero
    >>> print(ime.get_log()[3])
    exceptionStackTrace:
    >>> print(ime.get_log()[-1])
    END_ERROR
    """
    _error_1(logger, title, exception_type=exception,
             exception_value=str(exception),
             traceback=exception.__traceback__)


class _ProcessNoSS(_ProcessBase):
    """
    An internal class process implementation.

    This class implements a stand-alone process without explicit logging where
    the search and solution space are the same.
    """

    def __init__(self,
                 solution_space: Space,
                 objective: Objective,
                 algorithm: Algorithm,
                 log_file: Optional[Path] = None,
                 rand_seed: Optional[int] = None,
                 max_fes: Optional[int] = None,
                 max_time_millis: Optional[int] = None,
                 goal_f: Union[int, float, None] = None) -> None:
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
        """
        super().__init__(max_fes=max_fes,
                         max_time_millis=max_time_millis,
                         goal_f=goal_f)

        #: The solution space, i.e., the data structure of possible solutions.
        self._solution_space: Final[Space] = check_space(solution_space)
        #: The objective function rating candidate solutions.
        self.__objective: Final[Objective] = check_objective(objective)
        #: the internal invoker for the objective function
        self._f: Final[Callable] = self.__objective.evaluate
        #: The algorithm to be applied.
        self.__algorithm: Final[Algorithm] = check_algorithm(algorithm)
        #: The random seed.
        self.__rand_seed: Final[int] = rand_seed_generate() \
            if rand_seed is None \
            else rand_seed_check(rand_seed)
        #: The random number generator.
        self.__random: Final[Generator] = rand_generator(self.__rand_seed)
        #: The current best solution.
        self._current_best_y = solution_space.create()
        #: The current best objective value
        self._current_best_f: Union[int, float] = inf
        #: The log file, or `None` is needed
        if log_file is not None:
            if not isinstance(log_file, Path):
                raise type_error(log_file, "log_file", Path)
        self.__log_file: Final[Optional[Path]] = log_file
        #: the method for copying y
        self._copy_y: Final[Callable] = solution_space.copy
        #: set up the method forwards
        self.lower_bound = self.__objective.lower_bound  # type: ignore
        if self._end_f <= -inf:
            self._end_f = check_goal_f(self.lower_bound())
            self.lower_bound = lambda: self._end_f  # type: ignore
        self.upper_bound = objective.upper_bound  # type: ignore
        self.create = solution_space.create  # type: ignore
        self.copy = solution_space.copy  # type: ignore
        self.to_str = solution_space.to_str  # type: ignore
        self.is_equal = solution_space.is_equal  # type: ignore
        self.from_str = solution_space.from_str  # type: ignore
        self.n_points = solution_space.n_points  # type: ignore
        self.validate = solution_space.validate  # type: ignore
        #: the internal section logger
        self.__sections: Optional[Dict[str, str]] = \
            None if log_file is None else {}

    def get_random(self) -> Generator:
        return self.__random

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

    def has_best(self) -> bool:
        return self._current_fes > 0

    def get_best_f(self) -> Union[int, float]:
        if self._current_fes > 0:
            return self._current_best_f
        raise ValueError('No current best available.')

    def get_copy_of_best_x(self, x) -> None:
        if self._current_fes > 0:
            return self._copy_y(x, self._current_best_y)
        raise ValueError('No current best available.')

    def _log_own_parameters(self, logger: KeyValueLogSection) -> None:
        super()._log_own_parameters(logger)
        logger.key_value(logging.KEY_RAND_SEED, self.__rand_seed, True)
        logger.key_value(logging.KEY_RAND_GENERATOR_TYPE,
                         type_name_of(self.__random))
        logger.key_value(logging.KEY_RAND_BIT_GENERATOR_TYPE,
                         type_name_of(self.__random.bit_generator))

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        super().log_parameters_to(logger)
        with logger.scope(logging.SCOPE_ALGORITHM) as sc:
            self.__algorithm.log_parameters_to(sc)
        with logger.scope(logging.SCOPE_SOLUTION_SPACE) as sc:
            self._solution_space.log_parameters_to(sc)
        with logger.scope(logging.SCOPE_OBJECTIVE_FUNCTION) as sc:
            self.__objective.log_parameters_to(sc)

    def add_log_section(self, title: str, text: str) -> None:
        super().add_log_section(title, text)
        if self.__sections is not None:
            if title in self.__sections:
                raise ValueError(f"section '{title}' already logged.")
            self.__sections[title] = text.strip()

    def _write_log(self, logger: Logger) -> None:
        """Write the information gathered during optimization into the log."""
        with logger.key_values(logging.SECTION_FINAL_STATE) as kv:
            kv.key_value(logging.KEY_TOTAL_FES, self._current_fes)
            kv.key_value(logging.KEY_TOTAL_TIME_MILLIS,
                         _ns_to_ms(self._current_time_nanos
                                   - self._start_time_nanos))
            if self._current_fes > 0:
                kv.key_value(logging.KEY_BEST_F, self._current_best_f)
                kv.key_value(logging.KEY_LAST_IMPROVEMENT_FE,
                             self._last_improvement_fe)
                kv.key_value(logging.KEY_LAST_IMPROVEMENT_TIME_MILLIS,
                             _ns_to_ms(self._last_improvement_time_nanos
                                       - self._start_time_nanos))

        with logger.key_values(logging.SECTION_SETUP) as kv:
            self.log_parameters_to(kv)

        log_sys_info(logger)

        if self._current_fes > 0:
            with logger.text(logging.SECTION_RESULT_Y) as txt:
                txt.write(self._solution_space.to_str(self._current_best_y))

    def _validate_x(self) -> None:
        """Validate x, if it exists."""

    def _check_timing(self) -> None:
        """
        Check whether there has been any timing errors.

        :raises ValueError: if there is any timing error
        """
        if self._current_time_nanos < self._start_time_nanos:
            raise ValueError(
                f"current_time_nanos={self._current_time_nanos} < "
                f"start_time_nanos={self._start_time_nanos}")
        if self._current_fes <= 0:
            raise ValueError("no FE was performed")
        if self._current_fes < self._last_improvement_fe:
            raise ValueError(
                f"current_fe={self._current_fes} < "
                f"last_improvement_fe={self._last_improvement_fe}")
        if self._current_time_nanos < self._last_improvement_time_nanos:
            raise ValueError(
                f"current_time_nanos={self._current_time_nanos} < "
                "last_improvement_time_nanos="
                f"{self._last_improvement_time_nanos}")

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Exit the process and write the log if necessary."""
        # noinspection PyProtectedMember
        super().__exit__(exception_type, exception_value, traceback)

        # Update the total consumed time, but not include the error checks
        # below.
        self._current_time_nanos = _TIME_IN_NS()

        y_error: Optional[BaseException] = None  # error in solution?
        v_error: Optional[BaseException] = None  # error in objective value?
        x_error: Optional[BaseException] = None  # error in search space?
        t_error: Optional[BaseException] = None  # error in timing?
        log_error: Optional[BaseException] = None  # error while logging?
        try:
            self._solution_space.validate(self._current_best_y)
        except BaseException as be:
            y_error = be
        try:
            ff = self._f(self._current_best_y)
            if ff != self._current_best_f:
                raise ValueError(
                    "We re-computed the objective value of the best solution"
                    f"and got {ff}, but it has been registered as "
                    f"{self._current_best_f}!")
            if not isfinite(ff):
                raise ValueError(
                    f"Reproduced the objective value {ff} of the best "
                    "solution, but it is not finite?")
        except BaseException as be:
            v_error = be
        try:
            self._validate_x()
        except BaseException as be:
            x_error = be
        try:
            self._check_timing()
        except BaseException as be:
            t_error = be

        if self.__log_file is not None:
            with FileLogger(self.__log_file) as logger:
                try:
                    self._write_log(logger)
                except BaseException as be:
                    log_error = be

                if self._caught is not None:
                    _error_2(logger, logging.SECTION_ERROR_IN_RUN,
                             self._caught)
                if exception_type or exception_value or traceback:
                    _error_1(logger, logging.SECTION_ERROR_IN_CONTEXT,
                             exception_type, exception_value, traceback)
                if y_error:
                    _error_2(logger, logging.SECTION_ERROR_INVALID_Y, y_error)
                if v_error:
                    _error_2(logger, logging.SECTION_ERROR_BEST_F, v_error)
                if x_error:
                    _error_2(logger, logging.SECTION_ERROR_INVALID_X, x_error)
                if t_error:
                    _error_2(logger, logging.SECTION_ERROR_TIMING, t_error)
                if log_error:
                    _error_2(logger, logging.SECTION_ERROR_IN_LOG, log_error)

                # flush all the additional log sections at the end
                for t in sorted(self.__sections.keys()):
                    with logger.text(t) as sec:
                        sec.write(self.__sections[t])
                del self.__sections

        if not exception_type:
            # if no error happened when closing the process, raise any error
            # caught during validation.
            if self._caught is not None:
                raise self._caught  # pylint: disable=[E0702]
            if y_error:
                raise y_error
            if v_error:
                raise v_error
            if x_error:
                raise x_error
            if t_error:
                raise t_error
            if log_error:
                raise log_error

    def __str__(self) -> str:
        return "ProcessWithoutSearchSpace"
