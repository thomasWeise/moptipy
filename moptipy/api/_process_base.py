"""An internal module with the base class for implementing Processes."""
from abc import ABC
from math import inf
from threading import Lock, Timer
from time import monotonic_ns
from typing import Optional, Final, Union

from moptipy.api import logging
from moptipy.api.process import Process, check_goal_f, check_max_fes, \
    check_max_time_millis
from moptipy.utils.logger import KeyValueSection


class _ProcessBase(Process, ABC):
    """The internal base class for implementing optimization processes."""

    def __init__(self,
                 max_fes: Optional[int] = None,
                 max_time_millis: Optional[int] = None,
                 goal_f: Union[int, float, None] = None) -> None:
        """
        Initialize information that every black-box process must have.

        :param Optional[int] max_fes: the maximum permitted function
        evaluations
        :param Optional[int] max_time_millis: the maximum runtime in
        milliseconds
        :param Union[int, float, None] goal_f: the goal objective
        value: if it is reached, the process is terminated
        """
        super().__init__()
        #: The time when the process was started, in milliseconds.
        self._start_time_millis: Final[int] = int(monotonic_ns() // 1_000_000)
        #: The internal lock, needed to protect :meth:`terminate`.
        self.__lock: Final[Lock] = Lock()
        #: The maximum FEs.
        self._max_fes: Final[Optional[int]] = check_max_fes(max_fes, True)
        #: A version of :attr:`_max_fes` that can be used in comparisons.
        self._end_fes: Final[Union[int, float]] = inf \
            if (self._max_fes is None) else self._max_fes
        #: The goal objective value.
        self._goal_f: Final[Union[int, float, None]] = \
            check_goal_f(goal_f, True)
        #: A comparable version of :attr:`self._goal_f`.
        self._end_f: Final[Union[int, float]] = \
            -inf if (self._goal_f is None) else self._goal_f
        #: The currently consumed milliseconds.
        self._current_time_millis: int = 0
        #: The currently consumed objective function evaluations (FEs).
        self._current_fes: int = 0
        #: The time (in milliseconds) when the last improvement was made.
        self._last_improvement_time_millis: int = -1
        #: The FE when the last improvement was made.
        self._last_improvement_fe: int = -1
        #: The maximum runtime in milliseconds.
        self._max_time_millis: Final[Optional[int]] = \
            check_max_time_millis(max_time_millis, True)
        #: A comparable version of :attr:`_max_time_millis`.
        self._end_time_millis: Final[Union[float, int]] = \
            inf if (self._max_time_millis is None) else \
            int(self._start_time_millis + self._max_time_millis)
        #: The timer until the end-of-run, or `None` if there is no end time.
        self.__timer: Final[Optional[Timer]] = None \
            if (self._max_time_millis is None) else \
            Timer(interval=self._max_time_millis / 1_000.0,
                  function=self.terminate)

    def _after_init(self) -> None:
        """
        Finish initialization, start timer for termination if needed.

        Internal method that must be called after __init__ is completed.
        """
        if not (self.__timer is None):
            self.__timer.start()

    def get_consumed_fes(self) -> int:
        """
        Obtain the number consumed objective function evaluations.

        This is the number of calls to :meth:`evaluate`.

        :return: the number of objective function evaluations so far
        :rtype: int
        """
        return self._current_fes

    def get_consumed_time_millis(self) -> int:
        """
        Obtain an approximation of the consumed runtime in milliseconds.

        :return: the consumed runtime measured in milliseconds.
        :rtype: int
        """
        if not self._terminated:
            self._current_time_millis = int((monotonic_ns() + 999_999)
                                            // 1_000_000)
            if self._current_time_millis >= self._end_time_millis:
                self.terminate()
        return self._current_time_millis - self._start_time_millis

    def get_max_time_millis(self) -> Optional[int]:
        """
        Obtain the maximum runtime permitted in milliseconds.

        If no limit is set, `None` is returned.

        :return: the maximum runtime permitted in milliseconds,
        or `None` if no limit is specified.
        :rtype: Optional[int]
        """
        return self._max_time_millis

    def get_max_fes(self) -> Optional[int]:
        """
        Obtain the maximum number of permitted objective function evaluations.

        If no limit is set, `None` is returned.

        :return: the maximum number of objective function evaluations,
        or `None` if no limit is specified.
        :rtype: Optional[int]
        """
        return self._max_fes

    def get_last_improvement_fe(self) -> int:
        """
        Get the FE at which the last improvement was made.

        :return: the function evaluation when the last improvement was made
        :rtype: int
        :raises ValueError: if no FE was performed yet
        """
        if self._last_improvement_fe < 0:
            raise ValueError("Did not perform FE yet, cannot query "
                             "last improvement FE.")
        return self._last_improvement_fe

    def get_last_improvement_time_millis(self) -> int:
        """
        Get the FE at which the last improvement was made.

        :return: the function evaluation when the last improvement was made
        :rtype: int
        :raises ValueError: if no FE was performed yet
        """
        if self._last_improvement_time_millis < 0:
            raise ValueError("Did not perform FE yet, cannot query "
                             "last improvement time.")
        return self._last_improvement_time_millis - self._start_time_millis

    def _perform_termination(self) -> None:
        """
        Do the actual work of termination: must be invoked only once.

        This is an internal method invoked my :meth:`terminate`.
        """

    def terminate(self) -> None:
        """
        Terminate this process.

        This function is automatically called at the end of the `with`
        statement, but can also be called by the algorithm when it is
        finished and is also invoked automatically when a termination
        criterion is hit.
        After the first time this method is invoked, :meth:should_terminate`
        becomes `True`.

        While :meth:`terminate` can be called arbitrarily often, it is ensured
        that :meth:`_perform_termination` is called exactly once.
        """
        with self.__lock:
            old_terminated = self._terminated
            self._terminated = True
            if old_terminated:
                return
            if not (self.__timer is None):
                self.__timer.cancel()
            self._current_time_millis = int((monotonic_ns() + 999_999)
                                            // 1_000_000)
            self._perform_termination()

    def get_copy_of_current_best_y(self, y) -> None:
        """
        Get a copy of the current best point in the solution space.

        This method in this internal class just forwards to
        :meth:`get_copy_of_current_best_x`.

        :param y: the destination data structure to be overwritten
        """
        return self.get_copy_of_current_best_x(y)

    def _log_own_parameters(self, logger: KeyValueSection) -> None:
        """
        Write the parameters of this process to the logger.

        This includes the limits on runtime and FEs.

        :param Logger logger: the logger
        """
        super().log_parameters_to(logger)
        if not (self._max_fes is None):
            logger.key_value(logging.KEY_MAX_FES, self._max_fes)
        if not (self._max_time_millis is None):
            logger.key_value(logging.KEY_MAX_TIME_MILLIS,
                             self._max_time_millis)
        if not (self._goal_f is None):
            logger.key_value(logging.KEY_GOAL_F, self._goal_f)

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Write the standard parameters of this process to the logger.

        This includes the limits on runtime and FEs.

        :param Logger logger: the logger
        """
        with logger.scope(logging.SCOPE_PROCESS) as sc:
            self._log_own_parameters(sc)

    def get_name(self) -> str:
        """
        Get the name of this process implementation.

        :return: "baseProcess"
        """
        return "baseProcess"
