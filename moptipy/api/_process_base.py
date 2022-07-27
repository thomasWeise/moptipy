"""An internal module with the base class for implementing Processes."""
import os
from io import StringIO
from math import inf, isfinite
from threading import Lock, Timer
from time import time_ns
from traceback import print_tb
from typing import Optional, Union, Final, Callable, Dict, Any, List, cast

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm, check_algorithm
from moptipy.api.logging import _ALL_SECTIONS, KEY_MAX_FES, \
    KEY_MAX_TIME_MILLIS, SCOPE_PROCESS, KEY_GOAL_F, KEY_EXCEPTION_TYPE, \
    KEY_EXCEPTION_VALUE, KEY_EXCEPTION_STACK_TRACE, KEY_RAND_SEED, \
    KEY_RAND_GENERATOR_TYPE, KEY_RAND_BIT_GENERATOR_TYPE, SCOPE_ALGORITHM, \
    SCOPE_SOLUTION_SPACE, SCOPE_OBJECTIVE_FUNCTION, SECTION_FINAL_STATE, \
    KEY_TOTAL_FES, KEY_TOTAL_TIME_MILLIS, KEY_BEST_F, \
    KEY_LAST_IMPROVEMENT_FE, KEY_LAST_IMPROVEMENT_TIME_MILLIS, \
    SECTION_SETUP, SECTION_RESULT_Y, SECTION_ERROR_IN_RUN, \
    SECTION_ERROR_IN_CONTEXT, SECTION_ERROR_INVALID_Y, SECTION_ERROR_BEST_F, \
    SECTION_ERROR_INVALID_X, SECTION_ERROR_TIMING, SECTION_ERROR_IN_LOG
from moptipy.api.objective import Objective, check_objective
from moptipy.api.process import Process, check_max_fes, \
    check_max_time_millis
from moptipy.api.process import check_goal_f
from moptipy.api.space import Space, check_space
from moptipy.utils.logger import KeyValueLogSection, FileLogger, Logger, \
    SECTION_START, SECTION_END
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
                ts.write(KEY_EXCEPTION_TYPE)
                ts.write(": ")
                if isinstance(exception_type, str):
                    if exception_type.startswith("<class '"):
                        exception_type = exception_type[8:-2]
                    ts.write(exception_type.strip())
                else:
                    ts.write(type_name_of(exception_type))
                ts.write(os.linesep)
            if exception_value:
                ts.write(KEY_EXCEPTION_VALUE)
                ts.write(": ")
                ts.write(str(exception_value).strip())
                ts.write(os.linesep)
            if traceback:
                ts.write(KEY_EXCEPTION_STACK_TRACE)
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


# the function used to get the time
_TIME_IN_NS: Final[Callable[[], int]] = time_ns


def _ns_to_ms(nanos: int) -> int:
    """
    Convert nanoseconds to milliseconds by rounding up.

    :param int nanos: the nanoseconds
    :returns: the corresponding milliseconds, rounded up

    >>> _ns_to_ms(0)
    0
    >>> _ns_to_ms(1)
    1
    >>> _ns_to_ms(999_999)
    1
    >>> _ns_to_ms(1_000_000)
    1
    >>> _ns_to_ms(1_000_001)
    2
    >>> _ns_to_ms(1_999_999)
    2
    >>> _ns_to_ms(2_000_000)
    2
    >>> _ns_to_ms(2_000_001)
    3
    """
    return (nanos + 999_999) // 1_000_000


class _ProcessBase(Process):
    """The internal base class for implementing optimization processes."""

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
        super().__init__()
        #: This will be `True` after :meth:`terminate` has been called.
        self._terminated: bool = False
        #: This becomes `True` when :meth:`should_terminate` returned `True`.
        self._knows_that_terminated: bool = False
        #: The internal lock, needed to protect :meth:`terminate`.
        self.__lock: Final[Lock] = Lock()
        #: The maximum FEs.
        self._max_fes: Final[Optional[int]] = check_max_fes(max_fes, True)
        #: A version of :attr:`_max_fes` that can be used in comparisons.
        self._end_fes: Final[int] = 9_223_372_036_854_775_800 \
            if (self._max_fes is None) else self._max_fes
        #: The goal objective value.
        self._goal_f: Final[Union[int, float, None]] = \
            check_goal_f(goal_f, True)
        #: A comparable version of :attr:`self._goal_f`.
        self._end_f: Union[int, float] = \
            -inf if (self._goal_f is None) else self._goal_f
        #: The currently consumed nanoseconds.
        self._current_time_nanos: int = 0
        #: The currently consumed objective function evaluations (FEs).
        self._current_fes: int = 0
        #: The time (in nanoseconds) when the last improvement was made.
        self._last_improvement_time_nanos: int = -1
        #: The FE when the last improvement was made.
        self._last_improvement_fe: int = -1

        #: The solution space, i.e., the data structure of possible solutions.
        self._solution_space: Final[Space] = check_space(solution_space)
        #: The objective function rating candidate solutions.
        self.__objective: Final[Objective] = check_objective(objective)
        #: the internal invoker for the objective function
        self._f: Final[Callable[[Any], Union[int, float]]] = \
            self.__objective.evaluate
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
        self.is_always_integer = objective.is_always_integer  # type: ignore
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

        #: The time when the process was started, in nanoseconds.
        self._start_time_nanos: Final[int] = _TIME_IN_NS()
        #: The maximum runtime in milliseconds.
        self._max_time_millis: Final[Optional[int]] = \
            check_max_time_millis(max_time_millis, True)
        #: A comparable version of :attr:`_max_time_millis`, but representing
        #: the end time in nanoseconds rounded to the next highest
        #: millisecond.
        self._end_time_nanos: Final[Union[float, int]] = \
            inf if (self._max_time_millis is None) else \
            _ns_to_ms(int(self._start_time_nanos
                          + (1_000_000 * self._max_time_millis))) \
            * 1_000_000
        #: The timer until the end-of-run, or `None` if there is no end time.
        self.__timer: Final[Optional[Timer]] = None \
            if (self._max_time_millis is None) else \
            Timer(interval=self._max_time_millis / 1_000.0,
                  function=self.terminate)

        #: an internal base exception caught by the algorithm execution
        self._caught: Optional[BaseException] = None

    def _after_init(self) -> None:
        """
        Finish initialization, start timer for termination if needed.

        Internal method that must be called after __init__ is completed.
        """
        if self.__timer is not None:
            self.__timer.start()

    def get_random(self) -> Generator:
        return self.__random

    def should_terminate(self) -> bool:
        if self._terminated:
            self._knows_that_terminated = True
            return True
        return False

    def get_consumed_fes(self) -> int:
        return self._current_fes

    def get_consumed_time_millis(self) -> int:
        if not self._terminated:
            self._current_time_nanos = time = _TIME_IN_NS()
            if time >= self._end_time_nanos:
                self.terminate()
        return _ns_to_ms(self._current_time_nanos - self._start_time_nanos)

    def get_max_time_millis(self) -> Optional[int]:
        return self._max_time_millis

    def get_max_fes(self) -> Optional[int]:
        return self._max_fes

    def get_last_improvement_fe(self) -> int:
        if self._last_improvement_fe < 0:
            raise ValueError("Did not perform FE yet, cannot query "
                             "last improvement FE.")
        return self._last_improvement_fe

    def get_last_improvement_time_millis(self) -> int:
        if self._last_improvement_time_nanos < 0:
            raise ValueError("Did not perform FE yet, cannot query "
                             "last improvement time.")
        return _ns_to_ms(self._last_improvement_time_nanos
                         - self._start_time_nanos)

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

    def terminate(self) -> None:
        with self.__lock:
            old_terminated = self._terminated
            self._terminated = True
            if old_terminated:
                return
            if self.__timer is not None:
                self.__timer.cancel()
            del self.__timer
            self._current_time_nanos = _TIME_IN_NS()

    def get_copy_of_best_y(self, y) -> None:
        """
        Get a copy of the current best point in the solution space.

        This method in this internal class just forwards to
        :meth:`get_copy_of_best_x`.

        :param y: the destination data structure to be overwritten
        """
        return self.get_copy_of_best_x(y)

    def _log_own_parameters(self, logger: KeyValueLogSection) -> None:
        """
        Write the parameters of this process to the logger.

        This includes the limits on runtime and FEs.

        :param logger: the logger
        """
        super().log_parameters_to(logger)
        if not (self._max_fes is None):
            logger.key_value(KEY_MAX_FES, self._max_fes)
        if not (self._max_time_millis is None):
            logger.key_value(KEY_MAX_TIME_MILLIS,
                             self._max_time_millis)
        if not (self._goal_f is None):
            logger.key_value(KEY_GOAL_F, self._goal_f)
        logger.key_value(KEY_RAND_SEED, self.__rand_seed, True)
        logger.key_value(KEY_RAND_GENERATOR_TYPE,
                         type_name_of(self.__random))
        logger.key_value(KEY_RAND_BIT_GENERATOR_TYPE,
                         type_name_of(self.__random.bit_generator))

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Write the standard parameters of this process to the logger.

        This includes the limits on runtime and FEs.

        :param logger: the logger
        """
        with logger.scope(SCOPE_PROCESS) as sc:
            self._log_own_parameters(sc)
        with logger.scope(SCOPE_ALGORITHM) as sc:
            self.__algorithm.log_parameters_to(sc)
        with logger.scope(SCOPE_SOLUTION_SPACE) as sc:
            self._solution_space.log_parameters_to(sc)
        with logger.scope(SCOPE_OBJECTIVE_FUNCTION) as sc:
            self.__objective.log_parameters_to(sc)

    def add_log_section(self, title: str, text: str) -> None:
        """
        Add a section to the log, if a log is written (otherwise ignore it).

        :param title: the title of the log section
        :param text: the text to log
        """
        if not isinstance(title, str):
            raise type_error(title, "title", str)
        t = title.strip()
        if (len(t) != len(title)) or (len(t) <= 0) or (" " in t) \
                or ("\n" in t) or ("\t" in t):
            raise ValueError("section title must not be empty or contain "
                             f"white space, but '{title}' is/does.")
        if (t in _ALL_SECTIONS) or (SECTION_START in t) or (SECTION_END in t):
            raise ValueError(f"title '{t}' is a reserved section title")
        if t.upper() != t:
            raise ValueError("section titles must be in upper case,"
                             f"but yours is '{t}' (vs. '{t.upper()}'.")
        for ch in t:  # check all character codes in t
            code: int = ord(ch)  # we will only permit A-Z, 0-9, and _
            if not ((65 <= code <= 90) or (48 <= code <= 57) or (code == 95)):
                raise ValueError(
                    f"'{ch}' forbidden in section title, but got '{t}'.")
        if not isinstance(text, str):
            raise type_error(text, "text", str)
        if (SECTION_START in text) or (SECTION_END in text):
            raise ValueError(
                f"text of section '{t}' must not contain '{SECTION_START}' or"
                f" '{SECTION_END}' but is '{text}'")
        if self.__sections is not None:
            if title in self.__sections:
                raise ValueError(f"section '{title}' already logged.")
            self.__sections[title] = text.strip()

    def _log_best(self, kv: KeyValueLogSection) -> None:
        """
        Log the best solution.

        :param kv: the key values logger
        """
        kv.key_value(KEY_BEST_F, self._current_best_f)
        kv.key_value(KEY_LAST_IMPROVEMENT_FE,
                     self._last_improvement_fe)
        kv.key_value(KEY_LAST_IMPROVEMENT_TIME_MILLIS,
                     _ns_to_ms(self._last_improvement_time_nanos
                               - self._start_time_nanos))

    def _write_result(self, logger: Logger) -> None:
        """
        Write the end result into the log.

        :param logger: the logger
        """
        with logger.text(SECTION_RESULT_Y) as txt:
            txt.write(self._solution_space.to_str(self._current_best_y))

    def _write_log(self, logger: Logger) -> None:
        """Write the information gathered during optimization into the log."""
        with logger.key_values(SECTION_FINAL_STATE) as kv:
            kv.key_value(KEY_TOTAL_FES, self._current_fes)
            kv.key_value(KEY_TOTAL_TIME_MILLIS,
                         _ns_to_ms(self._current_time_nanos
                                   - self._start_time_nanos))
            if self._current_fes > 0:
                self._log_best(kv)

        with logger.key_values(SECTION_SETUP) as kv:
            self.log_parameters_to(kv)

        log_sys_info(logger)

        if self._current_fes > 0:
            self._write_result(logger)

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

    def _validate_best_f(self) -> None:
        """
        Validate the best encountered objective value.

        :raises ValueError: if there is an error
        """
        ff: Final[Union[int, float]] = self._f(self._current_best_y)
        if ff != self._current_best_f:
            raise ValueError(  # noqa
                "We re-computed the objective value of the best solution"
                f" and got {ff}, but it has been registered as "
                f"{self._current_best_f}!")  # noqa
        if not isfinite(ff):
            raise ValueError(  # noqa
                f"Reproduced the objective value {ff} of the best "
                "solution, but it is not finite?")  # noqa

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
        if self._current_fes > 0:
            try:
                self._validate_best_f()
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
                    _error_2(logger, SECTION_ERROR_IN_RUN,
                             self._caught)
                if exception_type or exception_value or traceback:
                    _error_1(logger, SECTION_ERROR_IN_CONTEXT,
                             exception_type, exception_value, traceback)
                if y_error:
                    _error_2(logger, SECTION_ERROR_INVALID_Y, y_error)
                if v_error:
                    _error_2(logger, SECTION_ERROR_BEST_F, v_error)
                if x_error:
                    _error_2(logger, SECTION_ERROR_INVALID_X, x_error)
                if t_error:
                    _error_2(logger, SECTION_ERROR_TIMING, t_error)
                if log_error:
                    _error_2(logger, SECTION_ERROR_IN_LOG, log_error)

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
        """
        Get the name of this process implementation.

        :return: "baseProcess"
        """
        return "baseProcess"


def _check_log_time(start_time: int, current_time: int,
                    log: List[List]) -> None:
    """
    Check the time inside the log.

    :param start_time: the start time
    :param current_time: the current time
    :param log: the log
    :raises ValueError: if there is a timing error in the log
    """
    last_time: int = -1
    last_fe: int = -1
    for row in log:
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
