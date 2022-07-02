"""The base classes for implementing optimization algorithms."""
from typing import Final

from moptipy.api.component import Component
from moptipy.api.logging import SCOPE_OP0, SCOPE_OP1, SCOPE_OP2
from moptipy.api.operators import Op0, check_op0, Op1, check_op1, \
    Op2, check_op2
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import PART_SEPARATOR
from moptipy.utils.types import type_error


# start book
class Algorithm(Component):
    """A base class for implementing optimization algorithms."""

    def solve(self, process: Process) -> None:
        """
        Apply this optimization algorithm to the given process.

        :param process: the process which provides methods to access the
            search space, the termination criterion, and a source of
            randomness. It also wraps the objective function, remembers the
            best-so-far solution, and takes care of creating log files (if
            this is wanted).
        """
# end book


class Algorithm0(Algorithm):
    """An algorithm with a nullary search operator."""

    def __init__(self, name: str, op0: Op0) -> None:
        """
        Create the algorithm with nullary search operator.

        :param name: the name of the algorithm
        :param op0: the nullary search operator
        """
        #: The nullary search operator.
        self.op0: Final[Op0] = check_op0(op0)
        if not isinstance(name, str):
            raise type_error(name, "name", str)
        if len(name) <= 0:
            raise ValueError(f"Algorithm name cannot be '{name}'.")
        self.name: Final[str] = name

    def __str__(self) -> str:
        """
        Get the name of the algorithm.

        :return: the name of the algorithm
        """
        return self.name

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_OP0) as sc:
            self.op0.log_parameters_to(sc)


class Algorithm1(Algorithm0):
    """An algorithm with a unary search operator."""

    def __init__(self, name: str, op0: Op0, op1: Op1) -> None:
        """
        Create the algorithm with nullary and unary search operator.

        :param name: the name of the algorithm
        :param op0: the nullary search operator
        :param op1: the unary search operator
        """
        super().__init__(name if op1.__class__ == Op1 else
                         f"{name}{PART_SEPARATOR}{op1}", op0)
        #: The unary search operator.
        self.op1: Final[Op1] = check_op1(op1)

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_OP1) as sc:
            self.op1.log_parameters_to(sc)


class Algorithm2(Algorithm1):
    """An algorithm with a binary and unary operator."""

    def __init__(self, name: str, op0: Op0, op1: Op1, op2: Op2) -> None:
        """
        Create the algorithm with nullary, unary, and binary search operator.

        :param name: the name of the algorithm
        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param op2: the binary search operator
        """
        super().__init__(
            name if op2.__class__ is Op2 else
            f"{name}{PART_SEPARATOR}{op2}", op0, op1)
        #: The binary search operator.
        self.op2: Final[Op2] = check_op2(op2)

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_OP2) as sc:
            self.op2.log_parameters_to(sc)


def check_algorithm(algorithm: Algorithm) -> Algorithm:
    """
    Check whether an object is a valid instance of :class:`Algorithm`.

    :param algorithm: the algorithm object
    :return: the object
    :raises TypeError: if `algorithm` is not an instance of :class:`Algorithm`
    """
    if not isinstance(algorithm, Algorithm):
        raise type_error(algorithm, "algorithm", Algorithm)
    return algorithm
