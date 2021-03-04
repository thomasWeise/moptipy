"""
A module providing a process without explicit logging where the search
and solution space are the same.
"""
from math import inf, isnan
from time import monotonic_ns
from typing import Optional, Union

from numpy.random import Generator

from moptipy.api._process_base import _ProcessBase
from moptipy.api.algorithm import Algorithm, _check_algorithm
from moptipy.api.objective import Objective, _check_objective
from moptipy.api.space import Space, _check_space
from moptipy.utils import logging
from moptipy.utils.logger import KeyValueSection, FileLogger, Logger
from moptipy.utils.nputils import rand_generator, rand_seed_generate, \
    rand_seed_check
from moptipy.utils.sys_info import log_sys_info


class _ProcessNoSS(_ProcessBase):
    """
    An internal class implementing a stand-alone process without explicit
    logging where the search and solution space are the same.
    """

    def __init__(self,
                 solution_space: Space,
                 objective: Objective,
                 algorithm: Algorithm,
                 log_file: str = None,
                 rand_seed: Optional[int] = None,
                 max_fes: Optional[int] = None,
                 max_time_millis: Optional[int] = None,
                 goal_f: Union[int, float, None] = None) -> None:

        super().__init__(max_fes=max_fes, max_time_millis=max_time_millis,
                         goal_f=goal_f)

        self._solution_space = _check_space(solution_space)
        self._objective = _check_objective(objective)
        self.__algorithm = _check_algorithm(algorithm)

        self.__rand_seed = rand_seed_generate() if rand_seed is None \
            else rand_seed_check(rand_seed)
        self.__random = rand_generator(self.__rand_seed)

        self._current_best_y = self._solution_space.create()
        self._current_best_f = inf
        self._has_current_best = False
        self.__log_file = log_file

    def get_random(self) -> Generator:
        return self.__random

    def lower_bound(self) -> Union[float, int]:
        return self._objective.lower_bound()

    def upper_bound(self) -> Union[float, int]:
        return self._objective.upper_bound()

    def evaluate(self, x) -> Union[float, int]:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError('The process has been terminated and '
                                 'the algorithm knows it.')
            return inf

        result = self._objective.evaluate(x)
        if isnan(result):
            raise ValueError("NaN invalid as objective value.")

        self._current_fes += 1

        do_term = self._current_fes >= self._end_fes

        if (self._current_fes <= 1) or (result < self._current_best_f):
            self._last_improvement_fe = self._current_fes
            self._current_best_f = result
            self._current_time_millis = int((monotonic_ns() + 999_999)
                                            // 1_000_000)
            self._last_improvement_time_millis = self._current_time_millis
            if self._current_time_millis >= self._end_time_millis:
                do_term = True
            self._solution_space.copy(x, self._current_best_y)
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
            return self._solution_space.copy(self._current_best_y, x)
        raise ValueError('No current best available.')

    def create(self):
        return self._solution_space.create()

    def copy(self, source, dest) -> None:
        self._solution_space.copy(source, dest)

    def to_str(self, x) -> str:
        return self._solution_space.to_str(x)

    def from_str(self, text: str):
        return self._solution_space.from_str(text)

    def is_equal(self, x1, x2) -> bool:
        return self._solution_space.is_equal(x1, x2)

    def validate(self, x) -> None:
        self._solution_space.validate(x)

    def scale(self) -> int:
        return self._solution_space.scale()

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_BBP_RAND_SEED, self.__rand_seed,
                         also_hex=True)
        logger.key_value(logging.KEY_BBP_RAND_GENERATOR_TYPE,
                         str(type(self.__random)))
        with logger.scope(logging.SCOPE_ALGORITHM) as sc:
            self.__algorithm.log_parameters_to(sc)
        with logger.scope(logging.SCOPE_SOLUTION_SPACE) as sc:
            self._solution_space.log_parameters_to(sc)
        with logger.scope(logging.SCOPE_OBJECTIVE_FUNCTION) as sc:
            self._objective.log_parameters_to(sc)

    def _write_log(self, logger: Logger) -> None:
        with logger.key_values(logging.SECTION_FINAL_STATE) as kv:
            kv.key_value(logging.KEY_ES_TOTAL_FES, self._current_fes)
            kv.key_value(logging.KEY_ES_TOTAL_TIME_MILLIS,
                         self._current_time_millis
                         - self._start_time_millis)
            if self._has_current_best:
                kv.key_value(logging.KEY_ES_BEST_F, self._current_best_f)
                kv.key_value(logging.KEY_ES_LAST_IMPROVEMENT_FE,
                             self._last_improvement_fe)
                kv.key_value(logging.KEY_ES_LAST_IMPROVEMENT_TIME_MILLIS,
                             self._last_improvement_time_millis
                             - self._start_time_millis)

        with logger.key_values(logging.SECTION_SETUP) as kv:
            self.log_parameters_to(kv)

        log_sys_info(logger)

        if self._has_current_best:
            with logger.text(logging.SECTION_RESULT_Y) as txt:
                txt.write(self._solution_space.to_str(self._current_best_y))

    def _perform_termination(self) -> None:
        # noinspection PyProtectedMember
        super()._perform_termination()
        if not (self.__log_file is None):
            with FileLogger(self.__log_file) as logger:
                self._write_log(logger)
            self.__log_file = None
        self._solution_space.validate(self._current_best_y)

    def log_state(self, key: str, value: Union[bool, int, float]) -> None:
        pass

    def get_name(self) -> str:
        return "ProcessWithoutSearchSpace"
