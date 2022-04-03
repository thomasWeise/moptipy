"""The algorithm execution API."""
from math import isfinite
from typing import Optional
from typing import Union

from moptipy.api._process_base import _ProcessBase
from moptipy.api._process_no_ss import _ProcessNoSS
from moptipy.api._process_no_ss_log import _ProcessNoSSLog
from moptipy.api._process_ss import _ProcessSS
from moptipy.api._process_ss_log import _ProcessSSLog
from moptipy.api.algorithm import Algorithm, check_algorithm
from moptipy.api.encoding import Encoding, check_encoding
from moptipy.api.objective import Objective, check_objective
from moptipy.api.process import Process
from moptipy.api.process import check_max_fes, check_max_time_millis, \
    check_goal_f
from moptipy.api.space import Space, check_space
from moptipy.utils.nputils import rand_seed_check
from moptipy.utils.path import Path


def _check_log_file(log_file: Optional[str],
                    none_is_ok: bool = True) -> Optional[Path]:
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
        self.__algorithm: Optional[Algorithm] = None
        self.__solution_space: Optional[Space] = None
        self.__objective: Optional[Objective] = None
        self.__search_space: Optional[Space] = None
        self.__encoding: Optional[Encoding] = None
        self.__rand_seed: Optional[int] = None
        self.__max_fes: Optional[int] = None
        self.__max_time_millis: Optional[int] = None
        self.__goal_f: Union[None, int, float] = None
        self.__log_file: Optional[Path] = None
        self.__log_improvements: bool = False
        self.__log_all_fes: bool = False

    def set_algorithm(self, algorithm: Algorithm) -> None:
        """
        Set the algorithm to be used for this experiment.

        :param algorithm: the algorithm
        """
        self.__algorithm = check_algorithm(algorithm)

    def get_algorithm(self) -> Algorithm:
        """
        Obtain the algorithm.

        Requires that :meth:`set_algorithm` was called first.

        :return: the algorithm
        """
        return check_algorithm(self.__algorithm)

    def set_solution_space(self, solution_space: Space) -> None:
        """
        Set the solution space to be used for this experiment.

        This is the space managing the data structure holding the candidate
        solutions.

        :param solution_space: the solution space
        """
        self.__solution_space = check_space(solution_space)

    def set_objective(self, objective: Objective) -> None:
        """
        Set the objective function to be used for this experiment.

        This is the function rating the quality of candidate solutions.

        :param objective: the objective function
        """
        self.__objective = check_objective(objective)

    def set_search_space(self, search_space: Optional[Space]) -> None:
        """
        Set the search space to be used for this experiment.

        This is the space from which the algorithm samples points.

        :param search_space: the search space, or `None` of none shall be
            used, i.e., if search and solution space are the same
        """
        self.__search_space = check_space(search_space, none_is_ok=True)

    def set_encoding(self, encoding: Optional[Encoding]) -> None:
        """
        Set the encoding to be used for this experiment.

        This is the function translating from the search space to the
        solution space.

        :param encoding: the encoding, or `None` of none shall be used
        """
        self.__encoding = check_encoding(encoding, none_is_ok=True)

    def set_rand_seed(self, rand_seed: Optional[int]) -> None:
        """
        Set the seed to be used for initializing the random number generator.

        :param rand_seed: the random seed, or `None` if a seed should
            automatically be chosen when the experiment is executed
        """
        self.__rand_seed = None if rand_seed is None \
            else rand_seed_check(rand_seed)

    def set_max_fes(self, max_fes: int,  # +book
                    force_override: bool = False) -> None:
        """
        Set the maximum FEs.

        This is the number of candidate solutions an optimization is allowed
        to evaluate. If this method is called multiple times, then the
        shortest limit is used unless `force_override` is `True`.

        :param max_fes: the maximum FEs
        :param force_override: the use the value given in `max_time_millis`
            regardless of what was specified before
        """
        max_fes = check_max_fes(max_fes)
        if not (self.__max_fes is None):
            if max_fes >= self.__max_fes:
                if not force_override:
                    return
        self.__max_fes = max_fes

    def set_max_time_millis(self, max_time_millis: int,
                            force_override: bool = False) -> None:
        """
        Set the maximum time in milliseconds.

        This is the maximum time that the process is allowed to run. If this
        method is called multiple times, the shortest time is used unless
        `force_override` is `True`.

        :param max_time_millis: the maximum time in milliseconds
        :param force_override: the use the value given in `max_time_millis`
            regardless of what was specified before
        """
        max_time_millis = check_max_time_millis(max_time_millis)
        if not (self.__max_time_millis is None):
            if max_time_millis >= self.__max_time_millis:
                if not force_override:
                    return
        self.__max_time_millis = max_time_millis

    def set_goal_f(self, goal_f: Union[int, float]) -> None:
        """
        Set the goal objective value after which the process can stop.

        If this method is called multiple times, then the largest value is
        retained.

        :param goal_f: the goal objective value.
        """
        goal_f = check_goal_f(goal_f)
        if not (self.__goal_f is None):
            if goal_f <= self.__goal_f:
                return
        self.__goal_f = goal_f

    def set_log_file(self, log_file: Optional[str]) -> None:
        """
        Set the log file to write to.

        This method can be called arbitrarily often.

        :param log_file: the log file
        """
        self.__log_file = _check_log_file(log_file, True)

    def set_log_improvements(self, log_improvements: bool = True) -> None:
        """
        Set whether improvements should be logged.

        :param log_improvements: if improvements should be logged?
        """
        if not isinstance(log_improvements, bool):
            raise ValueError("log improvements must be bool, but is "
                             f"{type(log_improvements)}.")
        self.__log_improvements = log_improvements

    def set_log_all_fes(self, log_all_fes: bool = True) -> None:
        """
        Set whether all FEs should be logged.

        :param log_all_fes: if all FEs should be logged?
        """
        if not isinstance(log_all_fes, bool):
            raise ValueError(
                f"log_all_FEs must be bool, but is {type(log_all_fes)}.")
        self.__log_all_fes = log_all_fes

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
        algorithm = check_algorithm(self.__algorithm)
        solution_space = check_space(self.__solution_space)
        objective = check_objective(self.__objective)
        search_space = check_space(self.__search_space,
                                   self.__encoding is None)
        encoding = check_encoding(self.__encoding,
                                  search_space is None)
        rand_seed = self.__rand_seed
        if not (rand_seed is None):
            rand_seed = rand_seed_check(rand_seed)
        max_time_millis = check_max_time_millis(self.__max_time_millis, True)
        max_fes = check_max_fes(self.__max_fes, True)
        goal_f = check_goal_f(self.__goal_f, True)
        f_lb = objective.lower_bound()
        if (not (f_lb is None)) and isfinite(f_lb):
            if (goal_f is None) or (f_lb > goal_f):
                goal_f = f_lb

        log_all_fes = self.__log_all_fes
        log_improvements = self.__log_improvements or self.__log_all_fes

        log_file = self.__log_file
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
