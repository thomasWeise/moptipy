"""The algorithm execution API."""
from math import isfinite
from typing import Final

from moptipy.api._process_base import _ProcessBase
from moptipy.api._process_no_ss import _ProcessNoSS
from moptipy.api._process_no_ss_log import _ProcessNoSSLog
from moptipy.api._process_ss import _ProcessSS
from moptipy.api._process_ss_log import _ProcessSSLog
from moptipy.api.algorithm import Algorithm, check_algorithm
from moptipy.api.encoding import Encoding, check_encoding
from moptipy.api.objective import Objective, check_objective
from moptipy.api.process import (
    Process,
    check_goal_f,
    check_max_fes,
    check_max_time_millis,
)
from moptipy.api.space import Space, check_space
from moptipy.utils.nputils import rand_seed_check
from moptipy.utils.path import Path
from moptipy.utils.types import type_error


def _check_log_file(log_file: str | None,
                    none_is_ok: bool = True) -> Path | None:
    """
    Check a log file.

    :param log_file: the log file
    :param none_is_ok: is `None` ok for log files?
    :return: the log file
    """
    if log_file is None:
        if none_is_ok:
            return None
    return Path.path(log_file)


class Execution:
    """
    Define all the components of an experiment and then execute it.

    This class follows the builder pattern. It allows us to
    step-by-step store all the parameters needed to execute an
    experiment. Via the method :meth:`~Execution.execute`, we can then
    run the experiment and obtain the instance of
    :class:`~moptipy.api.process.Process` *after* the execution of the
    algorithm. From this instance, we can query the final result of the
    algorithm application.
    """

    def __init__(self) -> None:
        """Initialize the execution builder."""
        super().__init__()
        self._algorithm: Algorithm | None = None
        self._solution_space: Space | None = None
        self._objective: Objective | None = None
        self._search_space: Space | None = None
        self._encoding: Encoding | None = None
        self._rand_seed: int | None = None
        self._max_fes: int | None = None
        self._max_time_millis: int | None = None
        self._goal_f: None | int | float = None
        self._log_file: Path | None = None
        self._log_improvements: bool = False
        self._log_all_fes: bool = False

    def set_algorithm(self, algorithm: Algorithm) -> "Execution":
        """
        Set the algorithm to be used for this experiment.

        :param algorithm: the algorithm
        :returns: this execution
        """
        self._algorithm = check_algorithm(algorithm)
        return self

    def set_solution_space(self, solution_space: Space) -> "Execution":
        """
        Set the solution space to be used for this experiment.

        This is the space managing the data structure holding the candidate
        solutions.

        :param solution_space: the solution space
        :returns: this execution
        """
        self._solution_space = check_space(solution_space)
        return self

    def set_objective(self, objective: Objective) -> "Execution":
        """
        Set the objective function to be used for this experiment.

        This is the function rating the quality of candidate solutions.

        :param objective: the objective function
        :returns: this execution
        """
        if self._objective is not None:
            raise ValueError(
                "Cannot add more than one objective function in single-"
                f"objective optimization, attempted to add {objective} "
                f"after {self._objective}.")
        self._objective = check_objective(objective)
        return self

    def set_search_space(self, search_space: Space | None) -> "Execution":
        """
        Set the search space to be used for this experiment.

        This is the space from which the algorithm samples points.

        :param search_space: the search space, or `None` of none shall be
            used, i.e., if search and solution space are the same
        :returns: this execution
        """
        self._search_space = check_space(search_space, none_is_ok=True)
        return self

    def set_encoding(self, encoding: Encoding | None) -> "Execution":
        """
        Set the encoding to be used for this experiment.

        This is the function translating from the search space to the
        solution space.

        :param encoding: the encoding, or `None` of none shall be used
        :returns: this execution
        """
        self._encoding = check_encoding(encoding, none_is_ok=True)
        return self

    def set_rand_seed(self, rand_seed: int | None) -> "Execution":
        """
        Set the seed to be used for initializing the random number generator.

        :param rand_seed: the random seed, or `None` if a seed should
            automatically be chosen when the experiment is executed
        """
        self._rand_seed = None if rand_seed is None \
            else rand_seed_check(rand_seed)
        return self

    def set_max_fes(self, max_fes: int,  # +book
                    force_override: bool = False) -> "Execution":
        """
        Set the maximum FEs.

        This is the number of candidate solutions an optimization is allowed
        to evaluate. If this method is called multiple times, then the
        shortest limit is used unless `force_override` is `True`.

        :param max_fes: the maximum FEs
        :param force_override: the use the value given in `max_time_millis`
            regardless of what was specified before
        :returns: this execution
        """
        max_fes = check_max_fes(max_fes)
        if self._max_fes is not None:
            if max_fes >= self._max_fes:
                if not force_override:
                    return self
        self._max_fes = max_fes
        return self

    def set_max_time_millis(self, max_time_millis: int,
                            force_override: bool = False) -> "Execution":
        """
        Set the maximum time in milliseconds.

        This is the maximum time that the process is allowed to run. If this
        method is called multiple times, the shortest time is used unless
        `force_override` is `True`.

        :param max_time_millis: the maximum time in milliseconds
        :param force_override: the use the value given in `max_time_millis`
            regardless of what was specified before
        :returns: this execution
        """
        max_time_millis = check_max_time_millis(max_time_millis)
        if self._max_time_millis is not None:
            if max_time_millis >= self._max_time_millis:
                if not force_override:
                    return self
        self._max_time_millis = max_time_millis
        return self

    def set_goal_f(self, goal_f: int | float) -> "Execution":
        """
        Set the goal objective value after which the process can stop.

        If this method is called multiple times, then the largest value is
        retained.

        :param goal_f: the goal objective value.
        :returns: this execution
        """
        goal_f = check_goal_f(goal_f)
        if self._goal_f is not None:
            if goal_f <= self._goal_f:
                return self
        self._goal_f = goal_f
        return self

    def set_log_file(self, log_file: str | None) -> "Execution":
        """
        Set the log file to write to.

        This method can be called arbitrarily often.

        :param log_file: the log file
        """
        self._log_file = _check_log_file(log_file, True)
        return self

    def set_log_improvements(self, log_improvements: bool = True) \
            -> "Execution":
        """
        Set whether improvements should be logged.

        :param log_improvements: if improvements should be logged?
        :returns: this execution
        """
        if not isinstance(log_improvements, bool):
            raise type_error(log_improvements, "log_improvements", bool)
        self._log_improvements = log_improvements
        return self

    def set_log_all_fes(self, log_all_fes: bool = True) -> "Execution":
        """
        Set whether all FEs should be logged.

        :param log_all_fes: if all FEs should be logged?
        :returns: this execution
        """
        if not isinstance(log_all_fes, bool):
            raise type_error(log_all_fes, "log_all_fes", bool)
        self._log_all_fes = log_all_fes
        return self

    def execute(self) -> Process:
        """
        Execute the experiment and return the process *after* the run.

        The optimization process constructed with this object is executed.
        This means that first, an instance of
        :class:`~moptipy.api.process.Process` is constructed.
        Then, the method :meth:`~moptipy.api.algorithm.Algorithm.solve` is
        applied to this instance.
        In other words, the optimization algorithm is executed until it
        terminates.
        Finally, this method returns the :class:`~moptipy.api.process.Process`
        instance *after* algorithm completion.
        This instance then can be queried for the final result of the run (via
        :meth:`~moptipy.api.process.Process.get_copy_of_best_y`), the
        objective value of this final best solution (via
        :meth:`~moptipy.api.process.Process.get_best_f`), and other
        information.

        :return: the process *after* the run, i.e., in the state where it can
            be queried for the result
        """
        algorithm: Final[Algorithm] = check_algorithm(self._algorithm)
        solution_space: Final[Space] = check_space(self._solution_space)
        objective: Final[Objective] = check_objective(self._objective)
        search_space: Final[Space | None] = check_space(
            self._search_space, self._encoding is None)
        encoding: Final[Encoding | None] = check_encoding(
            self._encoding, search_space is None)
        rand_seed = self._rand_seed
        if rand_seed is not None:
            rand_seed = rand_seed_check(rand_seed)
        max_time_millis = check_max_time_millis(self._max_time_millis, True)
        max_fes = check_max_fes(self._max_fes, True)
        goal_f = check_goal_f(self._goal_f, True)
        f_lb = objective.lower_bound()
        if (f_lb is not None) and isfinite(f_lb):
            if (goal_f is None) or (f_lb > goal_f):
                goal_f = f_lb

        log_all_fes = self._log_all_fes
        log_improvements = self._log_improvements or self._log_all_fes

        log_file = self._log_file
        if log_file is None:
            if log_all_fes:
                raise ValueError("Log file cannot be None "
                                 "if all FEs should be logged.")
            if log_improvements:
                raise ValueError("Log file cannot be None "
                                 "if improvements should be logged.")
        else:
            log_file.create_file_or_truncate()

        process: _ProcessBase
        if search_space is None:
            if log_improvements or log_all_fes:
                process = _ProcessNoSSLog(solution_space=solution_space,
                                          objective=objective,
                                          algorithm=algorithm,
                                          log_file=log_file,
                                          rand_seed=rand_seed,
                                          max_fes=max_fes,
                                          max_time_millis=max_time_millis,
                                          goal_f=goal_f,
                                          log_all_fes=log_all_fes)
            else:
                process = _ProcessNoSS(solution_space=solution_space,
                                       objective=objective,
                                       algorithm=algorithm,
                                       log_file=log_file,
                                       rand_seed=rand_seed,
                                       max_fes=max_fes,
                                       max_time_millis=max_time_millis,
                                       goal_f=goal_f)
        else:
            if log_improvements or log_all_fes:
                process = _ProcessSSLog(
                    solution_space=solution_space,
                    objective=objective,
                    algorithm=algorithm,
                    search_space=search_space,
                    encoding=encoding,
                    log_file=log_file,
                    rand_seed=rand_seed,
                    max_fes=max_fes,
                    max_time_millis=max_time_millis,
                    goal_f=goal_f,
                    log_all_fes=log_all_fes)
            else:
                process = _ProcessSS(solution_space=solution_space,
                                     objective=objective,
                                     algorithm=algorithm,
                                     search_space=search_space,
                                     encoding=encoding,
                                     log_file=log_file,
                                     rand_seed=rand_seed,
                                     max_fes=max_fes,
                                     max_time_millis=max_time_millis,
                                     goal_f=goal_f)
        try:
            # noinspection PyProtectedMember
            process._after_init()
            algorithm.solve(process)
        except BaseException as be:
            # noinspection PyProtectedMember
            process._caught = be
        return process
