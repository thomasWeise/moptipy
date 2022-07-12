"""The multi-objective algorithm execution API."""

from typing import Optional

from moptipy.api.execution import Execution
from moptipy.api.mo_problem import MOProblem, check_mo_problem, \
    MOSOProblemBridge
from moptipy.api.mo_process import MOProcess
from moptipy.api.objective import Objective, check_objective
from moptipy.utils.types import type_error


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
        self._archive_max_size: Optional[int] = None
        #: the archive size limit at which pruning should be performed
        self._archive_prune_limit: Optional[int] = None

    def set_archive_max_size(self, size: int) -> 'MOExecution':
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
        if not isinstance(size, int):
            raise type_error(size, "size", int)
        if size <= 0:
            raise ValueError(
                f"archive max size must be positive, but is {size}.")
        if self._archive_prune_limit is not None:
            if size > self._archive_prune_limit:
                raise ValueError(
                    f"archive max size {size} must be <= than archive "
                    f"prune limit {self._archive_prune_limit}")
        self._archive_max_size = size
        return self

    def set_objective(self, objective: Objective) -> 'MOExecution':
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

    def execute(self) -> MOProcess:
        """
        Create a multi-objective process, apply algorithm, and return result.

        This method is multi-objective equivalent of the
        :meth:`~moptipy.api.execution.Execution.execute` method. It returns a
        multi-objective process after applying the multi-objective algorithm.

        :returns: the instance of :class:`~moptipy.api.mo_process.MOProcess`
            after applying the algorithm.
        """
        return MOProcess()
