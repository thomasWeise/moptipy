from math import inf, isnan
from time import monotonic_ns
from typing import Optional, Union

from ._process_no_ss import _ProcessNoSS
from .component import Component
from .objective import Objective
from .space import Space
from ..utils import logging
from ..utils.logger import Logger


class _ProcessNoSSLog(_ProcessNoSS):

    def __init__(self,
                 solution_space: Space,
                 objective_function: Objective,
                 algorithm: Component,
                 log_file: str = None,
                 rand_seed: Optional[int] = None,
                 max_fes: Optional[int] = None,
                 max_time_millis: Optional[int] = None,
                 goal_f: Union[int, float, None] = None,
                 log_improvements: bool = False,
                 log_all_fes: bool = False):

        super().__init__(solution_space=solution_space,
                         objective_function=objective_function,
                         algorithm=algorithm,
                         log_file=log_file,
                         rand_seed=rand_seed,
                         max_fes=max_fes,
                         max_time_millis=max_time_millis,
                         goal_f=goal_f)
        if not isinstance(log_all_fes, bool):
            raise ValueError("log_all must be boolean, but is '" +
                             str(log_all_fes) + "'.")
        if not isinstance(log_improvements, bool):
            raise ValueError("log_improvements must be boolean, but is '" +
                             str(log_improvements) + "'.")
        self.__log_all = log_all_fes
        self.__log_improvements = log_improvements or log_all_fes
        self.__log = list()
        self.__log_header = [logging.PROGRESS_FES,
                             logging.PROGRESS_TIME_MILLIS,
                             logging.PROGRESS_CURRENT_F]
        self.__log_dict = {logging.PROGRESS_FES: 0,
                           logging.PROGRESS_TIME_MILLIS: 1,
                           logging.PROGRESS_CURRENT_F: 2}
        self.__last_res = None

    def evaluate(self, x) -> Union[float, int]:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError('The process has been terminated and the algorithm knows it.')
            return inf

        result = self._objective_function.evaluate(x)
        if isnan(result):
            raise ValueError("NaN invalid as objective value.")

        self.__last_res = result
        self._current_fes += 1

        do_term = self._current_fes >= self._end_fes
        do_log = self.__log_all

        if (self._current_fes <= 1) or (result < self._current_best_f):
            self._last_improvement_fe = self._current_fes
            self._current_best_f = result
            needs_time_millis = False
            self._current_time_millis = int((monotonic_ns() + 999_999) // 1_000_000)
            self._last_improvement_time_millis = self._current_time_millis
            if self._current_time_millis >= self._end_time_millis:
                do_term = True
            self._solution_space.x_copy(x, self._current_best_y)
            self._has_current_best = True
            do_log = do_log or self.__log_improvements
            if result <= self._end_f:
                do_term = True
        else:
            needs_time_millis = True

        if do_log and (not (self.__log is None)):
            if needs_time_millis:
                self._current_time_millis = int((monotonic_ns() + 999_999) // 1_000_000)
                if self._current_time_millis >= self._end_time_millis:
                    do_term = True
            self.__log.append([self._current_fes,
                               self._current_time_millis - self._start_time_millis,
                               result])

        if do_term:
            self.terminate()

    def log_state(self, key: str, value: Union[bool, int, float]):
        if self.__log is None:
            return

        list_len = len(self.__log_header)
        idx = self.__log_dict.setdefault(key, -1)
        if idx < 0:
            idx = list_len
            list_len += 1
            self.__log_dict[key] = idx
            self.__log_header.append(key)

        log_len = len(self.__log)
        if log_len > 0:
            entry = self.__log[log_len - 1]
            if entry[0] < self._current_fes:
                entry = None
            else:
                elen = len(entry)
                if elen < list_len:
                    entry += [None] * (list_len - elen)
        else:
            entry = None

        if entry is None:
            entry = [None] * list_len
            entry[0] = self._current_fes
            entry[2] = self.__last_res
            self.__log.append(entry)

        if (not (entry[idx] is None)) and (entry[idx] != value):
            ValueError("Entry for key'" + key
                       + "' already defined to be '"
                       + str(entry[idx]) +
                       "' and cannot be assigned again to '"
                       + str(value) + "'.")
        entry[idx] = value

    def _write_log(self, logger: Logger):
        if len(self.__log) > 0:
            list_len = len(self.__log_header)
            with logger.csv(logging.SECTION_PROGRESS, self.__log_header) as csv:
                for row in self.__log:
                    if len(row) < list_len:
                        row = row + [None] * (list_len - len(row))
                    csv.row(row)
        self.__log = None
        self.__log_header = None
        self.__log_dict = None
        super()._write_log(logger)

    def get_name(self):
        return "LoggingProcessWithoutSearchSpace"
