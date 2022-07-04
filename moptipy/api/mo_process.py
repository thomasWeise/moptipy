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
from typing import Union

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
