from math import inf, isnan
from time import monotonic_ns
from typing import Optional, Union

from ._process_no_ss import _ProcessNoSS
from .component import Component
from .objective import Objective
from .mapping import Mapping
from .space import Space
from ..utils.logger import KeyValues, Logger
from ..utils import logging


class _ProcessSS(_ProcessNoSS):
    def __init__(self,
                 solution_space: Space,
                 objective_function: Objective,
                 algorithm: Component,
                 log_file: str = None,
                 search_space: Space = None,
                 representation_mapping: Mapping = None,
                 rand_seed: Optional[int] = None,
                 max_fes: Optional[int] = None,
                 max_time_millis: Optional[int] = None,
                 goal_f: Union[int, float, None] = None):

        super().__init__(solution_space=solution_space,
                         objective_function=objective_function,
                         algorithm=algorithm,
                         log_file=log_file,
                         rand_seed=rand_seed,
                         max_fes=max_fes,
                         max_time_millis=max_time_millis,
                         goal_f=goal_f)

        if not isinstance(search_space, Space):
            raise ValueError("search_space should be instance of Space, but is "
                             + str(type(search_space)) + ".")
        self._search_space = search_space

        if not isinstance(representation_mapping, Mapping):
            raise ValueError("representation_mapping should be instance of Mapping, but is "
                             + str(type(representation_mapping)) + ".")
        self._representation_mapping = representation_mapping

        self._current_y = self._solution_space.x_create()
        self._current_best_x = self._search_space.x_create()

    def evaluate(self, x) -> Union[float, int]:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError('The process has been terminated and the algorithm knows it.')
            return inf

        self._representation_mapping.map(x, self._current_y)
        result = self._objective_function.evaluate(self._current_y)
        if isnan(result):
            raise ValueError("NaN invalid as objective value.")
        self._current_fes += 1

        do_term = self._current_fes >= self._end_fes

        if (self._current_fes <= 1) or (result < self._current_best_f):
            # noinspection PyAttributeOutsideInit
            self._last_improvement_fe = self._current_fes
            self._current_best_f = result
            self._search_space.x_copy(x, self._current_best_x)
            self._solution_space.x_copy(self._current_y, self._current_best_y)
            self._current_time_millis = int((monotonic_ns() + 999_999) // 1_000_000)
            self._last_improvement_time_millis = self._current_time_millis
            if self._current_time_millis >= self._end_time_millis:
                do_term = True

            # noinspection PyAttributeOutsideInit
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
            return self._search_space.x_copy(self._current_best_x, x)
        raise ValueError('No current best available.')

    def x_create(self):
        return self._search_space.x_create()

    def x_copy(self, source, dest):
        self._search_space.x_copy(source, dest)

    def x_to_str(self, x) -> str:
        return self._search_space.x_to_str(x)

    def x_is_equal(self, x1, x2) -> bool:
        return self._search_space.x_is_equal(x1, x2)

    def get_copy_of_current_best_y(self, y):
        return self._solution_space.x_copy(self._current_best_y, y)

    def log_parameters_to(self, logger: KeyValues):
        super().log_parameters_to(logger)
        with logger.scope(logging.SCOPE_SEARCH_SPACE) as sc:
            self._search_space.log_parameters_to(sc)
        with logger.scope(logging.SCOPE_REPRESENTATION_MAPPING) as sc:
            self._representation_mapping.log_parameters_to(sc)

    def _write_log(self, logger: Logger):
        # noinspection PyProtectedMember
        super()._write_log(logger)

        if self._has_current_best:
            with logger.text(logging.SECTION_RESULT_X) as txt:
                txt.write(self._search_space.x_to_str(self._current_best_x))

    def get_name(self):
        return "ProcessWithSearchSpace"
