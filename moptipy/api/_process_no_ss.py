"""Providing a process without explicit logging with a single space."""
from math import inf
from time import monotonic_ns
from typing import Optional, Union, Final, Callable

from numpy.random import Generator

from moptipy.api import logging
from moptipy.api._process_base import _ProcessBase
from moptipy.api.algorithm import Algorithm, check_algorithm
from moptipy.api.objective import Objective, check_objective
from moptipy.api.space import Space, check_space
from moptipy.utils.logger import KeyValueSection, FileLogger, Logger
from moptipy.utils.nputils import rand_generator, rand_seed_generate, \
    rand_seed_check
from moptipy.utils.path import Path
from moptipy.utils.sys_info import log_sys_info
from moptipy.utils.types import classname


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

        :param Space solution_space: the search- and solution space.
        :param Objective objective: the objective function
        :param Algorithm algorithm: the optimization algorithm
        :param Optional[Path] log_file: the optional log file
        :param Optional[int] rand_seed: the optional random seed
        :param Optional[int] max_fes: the maximum permitted function
            evaluations
        :param Optional[int] max_time_millis: the maximum runtime in
            milliseconds
        :param Union[int, float, None] goal_f: the goal objective
            value: if it is reached, the process is terminated
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
        self._current_best_y: Final = self._solution_space.create()
        #: The current best objective value
        self._current_best_f: Union[int, float] = inf
        #: Do we have a current-best solution?
        self._has_current_best: bool = False
        #: The log file, or `None` is needed
        self.__log_file: Final[Optional[Path]] = log_file
        #: the method for copying y
        self._copy_y: Final[Callable] = self._solution_space.copy
        #: set up the method forwards
        self.lower_bound = self.__objective.lower_bound  # type: ignore
        self.upper_bound = self.__objective.upper_bound  # type: ignore
        self.create = self._solution_space.create  # type: ignore
        self.copy = self._solution_space.copy  # type: ignore
        self.to_str = self._solution_space.to_str  # type: ignore
        self.is_equal = self._solution_space.is_equal  # type: ignore
        self.from_str = self._solution_space.from_str  # type: ignore
        self.n_points = self._solution_space.n_points  # type: ignore
        self.validate = self._solution_space.validate  # type: ignore

    def get_random(self) -> Generator:
        """
        Obtain the random number generator.

        :return: the random number generator
        :rtype: Generator
        """
        return self.__random

    def evaluate(self, x) -> Union[float, int]:
        """
        Evaluate a candidate solution.

        This method internally forwards to :meth:`Objective.evaluate` of
        :attr:`_objective` and keeps track of the best-so-far solution.

        :param x: the candidate solution
        :return: the objective value
        :rtype: Union[float, int]
        """
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError('The process has been terminated and '
                                 'the algorithm knows it.')
            return inf

        result: Final[Union[int, float]] = self._f(x)
        self._current_fes = current_fes = self._current_fes + 1
        do_term: bool = current_fes >= self._end_fes

        if (current_fes <= 1) or (result < self._current_best_f):
            self._last_improvement_fe = current_fes
            self._current_best_f = result
            self._current_time_millis = int((monotonic_ns() + 999_999)
                                            // 1_000_000)
            self._last_improvement_time_millis = self._current_time_millis
            if self._current_time_millis >= self._end_time_millis:
                do_term = True
            self._copy_y(self._current_best_y, x)
            self._has_current_best = True
            if result <= self._end_f:
                do_term = True

        if do_term:
            self.terminate()

        return result

    def has_current_best(self) -> bool:
        """
        Check whether a current best solution is available.

        As soon as one objective function evaluation has been performed,
        the black-box process can provide a best-so-far solution. Then,
        this method returns True. Otherwise, it returns False.

        :return: True if the current-best solution can be queried.
        :rtype: bool
        """
        return self._has_current_best

    def get_current_best_f(self) -> Union[int, float]:
        """
        Get the objective value of the current best solution.

        :return: the objective value of the current best solution.
        :rtype: Union[int,float]
        """
        if self._has_current_best:
            return self._current_best_f
        raise ValueError('No current best available.')

    def get_copy_of_current_best_x(self, x) -> None:
        """
        Get a copy of the current best point in the search space.

        :param x: the destination data structure to be overwritten
        """
        if self._has_current_best:
            return self._copy_y(x, self._current_best_y)
        raise ValueError('No current best available.')

    def _log_own_parameters(self, logger: KeyValueSection) -> None:
        super()._log_own_parameters(logger)
        logger.key_value(logging.KEY_RAND_SEED, self.__rand_seed,
                         also_hex=True)
        logger.key_value(logging.KEY_RAND_GENERATOR_TYPE,
                         classname(self.__random))
        logger.key_value(logging.KEY_RAND_BIT_GENERATOR_TYPE,
                         classname(self.__random.bit_generator))

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        super().log_parameters_to(logger)
        with logger.scope(logging.SCOPE_ALGORITHM) as sc:
            self.__algorithm.log_parameters_to(sc)
        with logger.scope(logging.SCOPE_SOLUTION_SPACE) as sc:
            self._solution_space.log_parameters_to(sc)
        with logger.scope(logging.SCOPE_OBJECTIVE_FUNCTION) as sc:
            self.__objective.log_parameters_to(sc)

    def _write_log(self, logger: Logger) -> None:
        with logger.key_values(logging.SECTION_FINAL_STATE) as kv:
            kv.key_value(logging.KEY_TOTAL_FES, self._current_fes)
            kv.key_value(logging.KEY_TOTAL_TIME_MILLIS,
                         self._current_time_millis
                         - self._start_time_millis)
            if self._has_current_best:
                kv.key_value(logging.KEY_BEST_F, self._current_best_f)
                kv.key_value(logging.KEY_LAST_IMPROVEMENT_FE,
                             self._last_improvement_fe)
                kv.key_value(logging.KEY_LAST_IMPROVEMENT_TIME_MILLIS,
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
        self._solution_space.validate(self._current_best_y)

    def get_name(self) -> str:
        return "ProcessWithoutSearchSpace"
