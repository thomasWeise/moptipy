"""An internal module with the base class for implementing Processes."""
from abc import ABC
from math import inf, isnan
from threading import Lock, Timer
from time import monotonic_ns
from typing import Optional
from typing import Union

from moptipy.api.process import Process
from moptipy.utils import logging
from moptipy.utils.logger import KeyValueSection


def _check_max_fes(max_fes: Optional[int],
                   none_is_ok: bool = False) -> Optional[int]:
    """
    Check the maximum FEs.
    :param Optional[int] max_fes: the maximum FEs
    :param bool none_is_ok: is None ok?
    :return: the maximum fes, or None
    :rtype: Optional[int]
    :raises TypeError: if max_fes is None or not an int
    :raises ValueError: if max_fes is invalid
    """
    if max_fes is None:
        if none_is_ok:
            return None
        raise TypeError("Maximum FEs must not be None.")

    if not isinstance(max_fes, int):
        raise TypeError("Maximum number of function evaluations must be "
                        "int, but is " + str(type(max_fes)) + ".")
    if max_fes <= 0:
        raise ValueError("Maximum FEs must be positive, but are "
                         + str(max_fes) + ".")
    return max_fes


def _check_max_time_millis(max_time_millis: Optional[int],
                           none_is_ok: bool = False) -> Optional[int]:
    """
    Check the maximum time in milliseconds.
    :param Optional[int] max_time_millis: the maximum time in milliseconds
    :param bool none_is_ok: is None ok?
    :return: the maximum time in millseconds, or None
    :rtype: Optional[int]
    :raises TypeError: if max_time_millis is None or not an int
    :raises ValueError: if max_time_millis is invalid
    """
    if max_time_millis is None:
        if none_is_ok:
            return None
        raise TypeError("Maximum time in milliseconds must not be None.")

    if not isinstance(max_time_millis, int):
        raise TypeError("Maximum time in milliseconds must be int, but is "
                        + str(type(max_time_millis)) + ".")
    if max_time_millis <= 0:
        raise ValueError("Maximum time in milliseconds must be positive, "
                         "but is " + str(max_time_millis) + ".")
    return max_time_millis


def _check_goal_f(goal_f: Union[int, float, None],
                  none_is_ok: bool = False) -> Union[int, float, None]:
    """
    Check the goal objective value.
    :param Optional[int] max_time_millis: the maximum time in milliseconds
    :param bool none_is_ok: is None ok?
    :return: the goal objective value, or None
    :rtype: Union[int, float, None]
    :raises TypeError: if goal_f is None or neither an int nor a float
    :raises ValueError: if goal_f is invalid
    """
    if goal_f is None:
        if none_is_ok:
            return None
        raise TypeError("Goal objective value cannot be None.")

    if not (isinstance(goal_f, (int, float))):
        raise TypeError("Goal objective value must be int or float, but is "
                        + str(type(goal_f)) + ".")
    if isnan(goal_f):
        raise ValueError("Goal objective value must not be NaN, but is"
                         + str(goal_f) + ".")
    if goal_f >= inf:
        raise ValueError("Goal objective value must be less than positive "
                         "infinity, but is " + str(goal_f) + ".")
    return goal_f


class _ProcessBase(Process, ABC):
    """
    This is the internal base class for implementing optimization processes.
    """

    def __init__(self,
                 max_fes: Optional[int] = None,
                 max_time_millis: Optional[int] = None,
                 goal_f: Union[int, float, None] = None) -> None:
        """
        Initialize information that every black-box process must have
        :param Optional[int] max_fes: the maximum permitted function
        evaluations
        :param Optional[int] max_time_millis: the maximum runtime in
        milliseconds
        :param Union[int, float, None] goal_f: the goal objective
        value: if it is reached, the process is terminated
        """
        super().__init__()
        self._start_time_millis: int = int(monotonic_ns() // 1_000_000)
        """The time when the process was started, in milliseconds."""
        self.__lock: Lock = Lock()
        """
        The internal lock, needed because terminate() may be invoked from
        another process.
        """

        self._max_fes = _check_max_fes(max_fes, True)
        if self._max_fes is None:
            self._end_fes = inf
        else:
            self._end_fes = self._max_fes

        self._goal_f = _check_goal_f(goal_f, True)
        if self._goal_f is None:
            self._end_f = -inf
        else:
            self._end_f = self._goal_f

        self._current_time_millis = 0
        """The currently consumed milliseconds."""
        self._current_fes = 0
        """The currently consumed objective function evaluations (FEs)."""
        self._last_improvement_time_millis = -1
        """The time (in milliseconds) when the last improvement was made."""
        self._last_improvement_fe = -1
        """The FE when the last improvement was made."""

        self._max_time_millis = _check_max_time_millis(max_time_millis, True)
        if self._max_time_millis is None:
            self._end_time_millis = inf
            self.__timer = None
        else:
            self._end_time_millis = int(self._start_time_millis
                                        + self._max_time_millis)
            self.__timer = Timer(interval=self._max_time_millis / 1_000.0,
                                 function=self.terminate)

    def _after_init(self) -> None:
        """This (internal) method must be called after __init__
        is completed."""
        if not (self.__timer is None):
            self.__timer.start()

    def get_consumed_fes(self) -> int:
        return self._current_fes

    def get_consumed_time_millis(self) -> int:
        if not self._terminated:
            self._current_time_millis = int((monotonic_ns() + 999_999)
                                            // 1_000_000)
            if self._current_time_millis >= self._end_time_millis:
                self.terminate()
        return self._current_time_millis - self._start_time_millis

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
        if self._last_improvement_time_millis < 0:
            raise ValueError("Did not perform FE yet, cannot query "
                             "last improvement time.")
        return self._last_improvement_time_millis - self._start_time_millis

    def _perform_termination(self) -> None:
        pass

    def terminate(self) -> None:
        # we guarantee that _perform_termination is called at most once
        with self.__lock:
            old_terminated = self._terminated
            self._terminated = True
            if old_terminated:
                return
            if not (self.__timer is None):
                self.__timer.cancel()
                self.__timer = None
            self._current_time_millis = int((monotonic_ns() + 999_999)
                                            // 1_000_000)
            self._perform_termination()

    def get_copy_of_current_best_y(self, y) -> None:
        return self.get_copy_of_current_best_x(y)

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        super().log_parameters_to(logger)
        if not (self._max_fes is None):
            logger.key_value(logging.KEY_BBP_MAX_FES, self._max_fes)
        if not (self._max_time_millis is None):
            logger.key_value(logging.KEY_BBP_MAX_TIME_MILLIS,
                             self._max_time_millis)
        if not (self._goal_f is None):
            logger.key_value(logging.KEY_BBP_GOAL_F, self._goal_f)

    def get_name(self) -> str:
        return "BaseProcess"
