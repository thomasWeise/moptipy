from math import inf, isnan
from typing import Final, Optional, Union
from time import monotonic_ns

from numpy.random import Generator
from numpy.random import default_rng

from ._process_base import _ProcessBase
from .component import Component
from .objective import Objective
from .space import Space
from ..utils import logging
from ..utils.logger import KeyValues, Logger


class _ProcessNoSS(_ProcessBase):
    __SEED_BYTES: Final = 8
    __MIN_RAND_SEED: Final = 0
    __MAX_RAND_SEED: Final = int((1 << (__SEED_BYTES * 8)) - 1)

    def __init__(self,
                 solution_space: Space,
                 objective_function: Objective,
                 algorithm: Component,
                 log_file: str = None,
                 rand_seed: Optional[int] = None,
                 max_fes: Optional[int] = None,
                 max_time_millis: Optional[int] = None,
                 goal_f: Union[int, float, None] = None):

        super().__init__(max_fes=max_fes, max_time_millis=max_time_millis, goal_f=goal_f)

        if not isinstance(solution_space, Space):
            raise ValueError("solution_space should be instance of Space, but is "
                             + str(type(solution_space)) + ".")
        self._solution_space = solution_space

        if not isinstance(objective_function, Objective):
            raise ValueError("objective_function should be instance of Objective, but is "
                             + str(type(objective_function)) + ".")
        self._objective_function = objective_function

        if not isinstance(algorithm, Component):
            raise ValueError("Algorithm must be instance of Component, but is instance of '"
                             + str(type(algorithm)) + "'.")
        self.__algorithm = algorithm

        if rand_seed is None:
            self.__rand_seed = int.from_bytes(default_rng().bytes(
                _ProcessNoSS.__SEED_BYTES),
                byteorder='big', signed=False)

        if not isinstance(self.__rand_seed, int):
            raise ValueError("rand_seed should be instance of int, but is "
                             + str(type(self.__rand_seed)) + ".")
        if (self.__rand_seed < _ProcessNoSS.__MIN_RAND_SEED) or \
                (self.__rand_seed > _ProcessNoSS.__MAX_RAND_SEED):
            raise ValueError("rand_seed must be in " +
                             str(_ProcessNoSS.__MIN_RAND_SEED) +
                             ".." +
                             str(_ProcessNoSS.__MAX_RAND_SEED))
        self.__random = default_rng(self.__rand_seed)

        self._current_best_y = self._solution_space.x_create()
        self._current_best_f = inf
        self._has_current_best = False
        self.__log_file = log_file

    def get_random(self) -> Generator:
        return self.__random

    def evaluate(self, x) -> Union[float, int]:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError('The process has been terminated and the algorithm knows it.')
            return inf

        result = self._objective_function.evaluate(x)
        if isnan(result):
            raise ValueError("NaN invalid as objective value.")

        self._current_fes += 1

        do_term = self._current_fes >= self._end_fes

        if (self._current_fes <= 1) or (result < self._current_best_f):
            self._last_improvement_fe = self._current_fes
            self._current_best_f = result
            self._current_time_millis = int((monotonic_ns() + 999_999) // 1_000_000)
            self._last_improvement_time_millis = self._current_time_millis
            if self._current_time_millis >= self._end_time_millis:
                do_term = True
            self._solution_space.x_copy(x, self._current_best_y)
            self._has_current_best = True
            if result <= self._end_f:
                do_term = True

        if do_term:
            self.terminate()

    def has_current_best(self) -> bool:
        return self._has_current_best

    def get_current_best_f(self) -> float:
        if self._has_current_best:
            return self._current_best_f
        raise ValueError('No current best available.')

    def get_copy_of_current_best_x(self, x):
        if self._has_current_best:
            return self._solution_space.x_copy(self._current_best_y, x)
        raise ValueError('No current best available.')

    def x_create(self):
        return self._solution_space.x_create()

    def x_copy(self, source, dest):
        self._solution_space.x_copy(source, dest)

    def x_to_str(self, x) -> str:
        return self._solution_space.x_to_str(x)

    def x_is_equal(self, x1, x2) -> bool:
        return self._solution_space.x_is_equal(x1, x2)

    def log_parameters_to(self, logger: KeyValues):
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_BBP_RAND_SEED, self.__rand_seed,
                         also_hex=True)
        logger.key_value(logging.KEY_BBP_RAND_GENERATOR_TYPE, str(type(self.__random)))
        with logger.scope(logging.SCOPE_ALGORITHM) as sc:
            self.__algorithm.log_parameters_to(sc)
        with logger.scope(logging.SCOPE_SOLUTION_SPACE) as sc:
            self._solution_space.log_parameters_to(sc)
        with logger.scope(logging.SCOPE_OBJECTIVE_FUNCTION) as sc:
            self._objective_function.log_parameters_to(sc)

    def _write_log(self, logger: Logger):
        with logger.key_values(logging.SECTION_FINAL_STATE) as kv:
            kv.key_value(logging.KEY_ES_TOTAL_FES, self._current_fes)
            kv.key_value(logging.KEY_ES_TOTAL_TIME_MILLIS,
                         self._current_time_millis -
                         self._start_time_millis)
            if self._has_current_best:
                kv.key_value(logging.KEY_ES_BEST_F, self._current_best_f)
                kv.key_value(logging.KEY_ES_LAST_IMPROVEMENT_FE,
                             self._last_improvement_fe)
                kv.key_value(logging.KEY_ES_LAST_IMPROVEMENT_TIME_MILLIS,
                             self._last_improvement_time_millis -
                             self._start_time_millis)

        with logger.key_values(logging.SECTION_SETUP) as kv:
            self.log_parameters_to(kv)

        if self._has_current_best:
            with logger.text(logging.SECTION_RESULT_Y) as txt:
                txt.write(self._solution_space.x_to_str(self._current_best_y))

    def _perform_termination(self):
        # noinspection PyProtectedMember
        super()._perform_termination()
        if not (self.__log_file is None):
            with Logger(self.__log_file) as logger:
                self._write_log(logger)
            self.__log_file = None

    def log_state(self, key: str, value: Union[bool, int, float]):
        pass

    def get_name(self):
        return "ProcessWithoutSearchSpace"
