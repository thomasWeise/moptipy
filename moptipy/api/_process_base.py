from abc import ABC
from math import inf, isnan
from threading import Lock, Timer
from time import monotonic_ns
from typing import Optional, Union

from .process import Process
from ..utils.logger import KeyValues
from ..utils import logging


class _ProcessBase(Process, ABC):

    def __init__(self,
                 max_fes: Optional[int] = None,
                 max_time_millis: Optional[int] = None,
                 goal_f: Union[int, float, None] = None):
        super().__init__()
        self._start_time_millis = int(monotonic_ns() // 1_000_000)
        self.lock = Lock()

        if max_fes is None:
            self._max_fes = None
            self._end_fes = inf
        else:
            self._max_fes = int(max_fes)
            if self._max_fes <= 0:
                raise ValueError("Maximum FEs must be positive, but are "
                                 + str(self._max_fes) + ".")
            self._end_fes = self._max_fes

        if goal_f is None:
            self._goal_f = None
            self._end_f = -inf
        else:
            if isinstance(goal_f, int) or isinstance(goal_f, float):
                self._goal_f = goal_f
            else:
                self._goal_f = float(goal_f)
            if isnan(self._goal_f):
                raise ValueError("Goal F must not be NaN, but is"
                                 + str(self._goal_f) + ".")
            self._end_f = self._goal_f

        self._current_time_millis = 0
        self._current_fes = 0
        self._last_improvement_time_millis = -1
        self._last_improvement_fe = -1

        if max_time_millis is None:
            self._max_time_millis = None
            self._end_time_millis = inf
            self._timer = None
        else:
            self._max_time_millis = int(max_time_millis)
            if self._max_time_millis <= 0:
                raise ValueError("Maximum time in milliseconds must be positive, but is "
                                 + str(self._max_time_millis) + ".")
            self._end_time_millis = int(self._start_time_millis + self._max_time_millis)
            self._timer = Timer(interval=self._max_time_millis / 1_000.0,
                                function=self.terminate)

    def _after_init(self):
        """This (internal) method must be called after the __init__ method is completed."""
        if not (self._timer is None):
            self._timer.start()

    def get_consumed_fes(self) -> int:
        return self._current_fes

    def get_consumed_time_millis(self) -> int:
        if not self._terminated:
            self._current_time_millis = int((monotonic_ns() + 999_999) // 1_000_000)
            if self._current_time_millis >= self._end_time_millis:
                self.terminate()
        return self._current_time_millis - self._start_time_millis

    def get_max_time_millis(self) -> Optional[int]:
        return self._max_time_millis

    def get_max_fes(self) -> Optional[int]:
        return self._max_fes

    def get_last_improvement_fe(self) -> int:
        if self._last_improvement_fe < 0:
            raise ValueError("Did not perform FE yet, cannot query last improvement FE.")
        return self._last_improvement_fe

    def get_last_improvement_time_millis(self) -> int:
        if self._last_improvement_time_millis < 0:
            raise ValueError("Did not perform FE yet, cannot query last improvement time.")
        return self._last_improvement_time_millis - self._start_time_millis

    def _perform_termination(self):
        pass

    def terminate(self):
        # we guarantee that _perform_termination is called at most once
        with self.lock:
            old_terminated = self._terminated
            self._terminated = True
            if old_terminated:
                return
            if not (self._timer is None):
                self._timer.cancel()
                self._timer = None
            self._current_time_millis = int((monotonic_ns() + 999_999) // 1_000_000)
            self._perform_termination()

    def get_copy_of_current_best_y(self, y):
        return self.get_copy_of_current_best_x(y)

    def log_parameters_to(self, logger: KeyValues):
        super().log_parameters_to(logger)
        if not (self._max_fes is None):
            logger.key_value(logging.KEY_BBP_MAX_FES, self._max_fes)
        if not (self._max_time_millis is None):
            logger.key_value(logging.KEY_BBP_MAX_TIME_MILLIS, self._max_time_millis)
        if not (self._goal_f is None):
            logger.key_value(logging.KEY_BBP_GOAL_F, self._goal_f)

    def get_name(self):
        return "BaseProcess"
