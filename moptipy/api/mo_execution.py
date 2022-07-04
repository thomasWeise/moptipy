"""The multi-objective algorithm execution API."""

from moptipy.api.execution import Execution
from moptipy.api.mo_process import MOProcess


class MOExecution(Execution):
    """
    Define all the components of a multi-objective experiment and execute it.

    Different from :class:`~moptipy.api.execution.Execution`, this class here
    allows us to construct multi-objective optimization processes, i.e., such
    that have more than one optimization goal.
    """

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
