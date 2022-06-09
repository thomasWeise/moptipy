"""An internal module with the base class for implementing Processes."""
from math import inf
from threading import Lock, Timer
from time import time_ns
from typing import Optional, Final, Union, Callable

from moptipy.api.logging import _ALL_SECTIONS, KEY_MAX_FES, \
    KEY_MAX_TIME_MILLIS, SCOPE_PROCESS, KEY_GOAL_F
from moptipy.api.process import Process, check_goal_f, check_max_fes, \
    check_max_time_millis
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.logger import SECTION_START, SECTION_END
from moptipy.utils.types import type_error

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
                 max_fes: Optional[int] = None,
                 max_time_millis: Optional[int] = None,
                 goal_f: Union[int, float, None] = None) -> None:
        """
        Initialize information that every black-box process must have.

        :param max_fes: the maximum permitted function evaluations
        :param max_time_millis: the maximum runtime in milliseconds
        :param goal_f: the goal objective value. if it is reached, the process
            is terminated
        """
        super().__init__()
        #: This will be `True` after :meth:`terminate` has been called.
        self._terminated: bool = False  # +book
        #: This becomes `True` when :meth:`should_terminate` returned `True`.
        self._knows_that_terminated: bool = False
        #: The time when the process was started, in nanoseconds.
        self._start_time_nanos: Final[int] = _TIME_IN_NS()
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

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Write the standard parameters of this process to the logger.

        This includes the limits on runtime and FEs.

        :param logger: the logger
        """
        with logger.scope(SCOPE_PROCESS) as sc:
            self._log_own_parameters(sc)

    def __str__(self) -> str:
        """
        Get the name of this process implementation.

        :return: "baseProcess"
        """
        return "baseProcess"

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
            raise ValueError("Section title must not be empty or contain "
                             f"white space, but '{title}' is/does.")
        if (t in _ALL_SECTIONS) or (SECTION_START in t) or (SECTION_END in t):
            raise ValueError(f"title '{t}' is a forbidden section title")
        if t.upper() != t:
            raise ValueError("section titles must be in upper case,"
                             f"but yours is '{t}' (vs. '{t.upper()}'.")
        if not isinstance(text, str):
            raise type_error(text, "text", str)
        if (SECTION_START in t) or (SECTION_END in t):
            raise ValueError(
                f"text of section '{t}' must not contain '{SECTION_START}' or"
                f" '{SECTION_END}' but is '{text}'")
