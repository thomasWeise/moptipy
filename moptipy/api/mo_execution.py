"""The multi-objective algorithm execution API."""

from math import isfinite
from typing import Final, cast

from moptipy.api._mo_process_no_ss import _MOProcessNoSS
from moptipy.api._mo_process_no_ss_log import _MOProcessNoSSLog
from moptipy.api._mo_process_ss import _MOProcessSS
from moptipy.api._mo_process_ss_log import _MOProcessSSLog
from moptipy.api.algorithm import Algorithm, check_algorithm
from moptipy.api.encoding import Encoding, check_encoding
from moptipy.api.execution import Execution
from moptipy.api.mo_archive import MOArchivePruner, check_mo_archive_pruner
from moptipy.api.mo_problem import (
    MOProblem,
    MOSOProblemBridge,
    check_mo_problem,
)
from moptipy.api.mo_process import MOProcess
from moptipy.api.objective import Objective, check_objective
from moptipy.api.process import (
    check_goal_f,
    check_max_fes,
    check_max_time_millis,
)
from moptipy.api.space import Space, check_space
from moptipy.mo.archive.keep_farthest import KeepFarthest
from moptipy.utils.nputils import rand_seed_check
from moptipy.utils.types import check_int_range


class MOExecution(Execution):
    """
    Define all the components of a multi-objective experiment and execute it.

    Different from :class:`~moptipy.api.execution.Execution`, this class here
    allows us to construct multi-objective optimization processes, i.e., such
    that have more than one optimization goal.
    """

    def __init__(self) -> None:
        """Create the multi-objective execution."""
        super().__init__()
        #: the maximum size of a pruned archive
        self._archive_max_size: int | None = None
        #: the archive size limit at which pruning should be performed
        self._archive_prune_limit: int | None = None
        #: the archive pruning strategy
        self._archive_pruner: MOArchivePruner | None = None

    def set_archive_max_size(self, size: int) -> "MOExecution":
        """
        Set the upper limit for the archive size (after pruning).

        The internal archive of the multi-objective optimization process
        retains non-dominated solutions encountered during the search. Since
        there can be infinitely many such solutions, the archive could grow
        without bound if left untouched.
        Therefore, we define two size limits: the maximum archive size
        (defined by this method) and the pruning limit. Once the archive grows
        beyond the pruning limit, it is cut down to the archive size limit.

        :param size: the maximum archive size
        :returns: this execution
        """
        check_int_range(size, "maximum archive size")
        if (self._archive_prune_limit is not None) and \
                (size > self._archive_prune_limit):
            raise ValueError(
                f"archive max size {size} must be <= than archive "
                f"prune limit {self._archive_prune_limit}")
        self._archive_max_size = size
        return self

    def set_archive_pruning_limit(self, limit: int) -> "MOExecution":
        """
        Set the size limit of the archive above which pruning is performed.

        If the size of the archive grows above this limit, the archive will be
        pruned down to the archive size limit.

        :param limit: the archive pruning limit
        :returns: this execution
        """
        check_int_range(limit, "limit", 1)
        if (self._archive_max_size is not None) and \
                (limit < self._archive_max_size):
            raise ValueError(
                f"archive pruning limit {limit} must be >= than archive "
                f"maximum size {self._archive_max_size}")
        self._archive_prune_limit = limit
        return self

    def set_archive_pruner(self, pruner: MOArchivePruner) -> "MOExecution":
        """
        Set the pruning strategy for downsizing the archive.

        :param pruner: the archive pruner
        :returns: this execution
        """
        self._archive_pruner = check_mo_archive_pruner(pruner)
        return self

    def set_objective(self, objective: Objective) -> "MOExecution":
        """
        Set the objective function in form of a multi-objective problem.

        :param objective: the objective function
        :returns: this execution
        """
        check_objective(objective)
        if not isinstance(objective, MOProblem):
            objective = MOSOProblemBridge(objective)
        super().set_objective(check_mo_problem(objective))
        return self

    def set_algorithm(self, algorithm: Algorithm) -> "MOExecution":
        """
        Set the algorithm to be used for this experiment.

        :param algorithm: the algorithm
        :returns: this execution
        """
        super().set_algorithm(algorithm)
        return self

    def set_solution_space(self, solution_space: Space) -> "MOExecution":
        """
        Set the solution space to be used for this experiment.

        This is the space managing the data structure holding the candidate
        solutions.

        :param solution_space: the solution space
        :returns: this execution
        """
        super().set_solution_space(solution_space)
        return self

    def set_search_space(self, search_space: Space | None) -> "MOExecution":
        """
        Set the search space to be used for this experiment.

        This is the space from which the algorithm samples points.

        :param search_space: the search space, or `None` of none shall be
            used, i.e., if search and solution space are the same
        :returns: this execution
        """
        super().set_search_space(search_space)
        return self

    def set_encoding(self, encoding: Encoding | None) -> "MOExecution":
        """
        Set the encoding to be used for this experiment.

        This is the function translating from the search space to the
        solution space.

        :param encoding: the encoding, or `None` of none shall be used
        :returns: this execution
        """
        super().set_encoding(encoding)
        return self

    def set_rand_seed(self, rand_seed: int | None) -> "MOExecution":
        """
        Set the seed to be used for initializing the random number generator.

        :param rand_seed: the random seed, or `None` if a seed should
            automatically be chosen when the experiment is executed
        """
        super().set_rand_seed(rand_seed)
        return self

    def set_max_fes(self, max_fes: int,
                    force_override: bool = False) -> "MOExecution":
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
        super().set_max_fes(max_fes, force_override)
        return self

    def set_max_time_millis(self, max_time_millis: int,
                            force_override: bool = False) -> "MOExecution":
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
        super().set_max_time_millis(max_time_millis, force_override)
        return self

    def set_goal_f(self, goal_f: int | float) -> "MOExecution":
        """
        Set the goal objective value after which the process can stop.

        If this method is called multiple times, then the largest value is
        retained.

        :param goal_f: the goal objective value.
        :returns: this execution
        """
        super().set_goal_f(goal_f)
        return self

    def set_log_file(self, log_file: str | None) -> "MOExecution":
        """
        Set the log file to write to.

        This method can be called arbitrarily often.

        :param log_file: the log file
        """
        super().set_log_file(log_file)
        return self

    def set_log_improvements(self, log_improvements: bool = True) \
            -> "MOExecution":
        """
        Set whether improvements should be logged.

        :param log_improvements: if improvements should be logged?
        :returns: this execution
        """
        super().set_log_improvements(log_improvements)
        return self

    def set_log_all_fes(self, log_all_fes: bool = True) -> "MOExecution":
        """
        Set whether all FEs should be logged.

        :param log_all_fes: if all FEs should be logged?
        :returns: this execution
        """
        super().set_log_all_fes(log_all_fes)
        return self

    def execute(self) -> MOProcess:
        """
        Create a multi-objective process, apply algorithm, and return result.

        This method is multi-objective equivalent of the
        :meth:`~moptipy.api.execution.Execution.execute` method. It returns a
        multi-objective process after applying the multi-objective algorithm.

        :returns: the instance of :class:`~moptipy.api.mo_process.MOProcess`
            after applying the algorithm.
        """
        objective: Final[MOProblem] = cast(MOProblem, self._objective)
        solution_space: Final[Space] = check_space(self._solution_space)
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
        if (f_lb is not None) and isfinite(f_lb) and \
                ((goal_f is None) or (f_lb > goal_f)):
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

        pruner: Final[MOArchivePruner] = \
            self._archive_pruner if self._archive_pruner is not None \
            else KeepFarthest(objective)
        dim: Final[int] = objective.f_dimension()
        size: Final[int] = self._archive_max_size if \
            self._archive_max_size is not None else (
            self._archive_prune_limit if
            self._archive_prune_limit is not None
            else (1 if dim == 1 else 32))
        limit: Final[int] = self._archive_prune_limit if \
            self._archive_prune_limit is not None \
            else (1 if dim == 1 else (size * 4))
        algorithm: Final[Algorithm] = check_algorithm(self._algorithm)

        process: Final[_MOProcessNoSS] = (_MOProcessNoSSLog(
            solution_space, objective, algorithm, pruner, size, limit,
            log_file, rand_seed, max_fes, max_time_millis, goal_f,
            log_all_fes) if log_improvements or log_all_fes else
            _MOProcessNoSS(solution_space, objective, algorithm, pruner,
                           size, limit, log_file, rand_seed, max_fes,
                           max_time_millis, goal_f)) \
            if search_space is None else (_MOProcessSSLog(
                solution_space, objective, algorithm, pruner, size, limit,
                log_file, search_space, encoding, rand_seed, max_fes,
                max_time_millis, goal_f,
                log_all_fes) if log_improvements or log_all_fes else
            _MOProcessSS(solution_space, objective, algorithm, pruner, size,
                         limit, log_file, search_space, encoding, rand_seed,
                         max_fes, max_time_millis, goal_f))

        try:
            # noinspection PyProtectedMember
            process._after_init()  # finalize the created process
            pruner.initialize()  # initialize the pruner
            objective.initialize()  # initialize the multi-objective problem
            if encoding is not None:
                encoding.initialize()   # initialize the encoding
            solution_space.initialize()  # initialize the solution space
            if search_space is not None:
                search_space.initialize()  # initialize the search space
            algorithm.initialize()  # initialize the algorithm
            algorithm.solve(process)  # apply the algorithm
        except Exception as be:
            # noinspection PyProtectedMember
            process._caught = be
        return process
