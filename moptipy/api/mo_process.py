"""
Processes offer data to both the user and the optimization algorithm.

They provide the information about the optimization process and its current
state as handed to the optimization algorithm and, after the algorithm has
finished, to the user.
This is the multi-objective version of the :class:`~moptipy.api.process.\
Process`-API.
It supports having multiple objective functions.
It also provides a single core objective value, which is the scalarized
result of several objective values.
"""
from typing import Union, List, Any

import numpy as np

from moptipy.api.mo_archive import MORecord
from moptipy.api.mo_problem import MOProblem
from moptipy.api.process import Process


class MOProcess(MOProblem, Process):
    """
    A multi-objective :class:`~moptipy.api.process.Process` API variant.

    This class encapsulates an optimization process using multiple objective
    functions. It inherits all of its methods from the single-objective
    process :class:`~moptipy.api.process.Process` and extends its API towards
    multi-objective optimization by providing the functionality of the class
    :class:`~moptipy.api.mo_problem.MOProblem`.
    """

    def register(self, x, f: Union[int, float]) -> None:
        """Unavailable during multi-objective optimization."""
        raise ValueError(
            "register is not available during multi-objective optimization.")

    def get_archive(self) -> List[MORecord]:
        """
        Get the archive of non-dominated solutions.

        :returns: a list containing all non-dominated solutions currently in
            the archive
        """

    def check_in(self, x: Any, fs: np.ndarray,
                 prune_if_necessary: bool = False) -> bool:
        """
        Check a solution into the archive.

        This method is intended for being invoked after the optimization
        algorithm has finished its work. The algorithm should, by itself,
        maintain the set of interesting solutions during its course. Once
        it has completed all of its computations, it should flush these
        solutions to the process using this method. All the non-dominated
        solutions preserved in the archive will then become available via
        :meth:`get_archive` to the code starting the optimization procedure.

        :param x: the point in the search space
        :param fs: the vector of objective values
        :param prune_if_necessary: should we prune the archive if it becomes
            too large? `False` means that the archive may grow unbounded
        :returns: `True` if the solution was non-dominated and has actually
            entered the archive, `False` if it has not entered the archive
        """

    def __str__(self) -> str:
        """
        Get the name of this process implementation.

        :return: "mo_process"
        """
        return "mo_process"

    def __enter__(self) -> 'MOProcess':
        """
        Begin a `with` statement.

        :return: this process itself
        """
        return self
