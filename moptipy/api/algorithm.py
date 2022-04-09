"""The base classes for implementing optimization algorithms."""
from typing import Callable, Final

from moptipy.api.component import Component, CallableComponent
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

    def __init__(self,
                 op0: Op0,
                 op0_is_default: bool = True) -> None:
        """
        Create the algorithm with nullary search operator.

        :param op0: the nullary search operator
        :param op0_is_default: is this a default nullary operator?
            if `True`, it will not be included in the name suffix
        """
        #: The nullary search operator.
        self.op0: Final[Op0] = check_op0(op0)

        if not isinstance(op0_is_default, bool):
            raise type_error(op0_is_default, "op0_is_default", bool)
        #: The internal name suffix
        self._name_suffix: str = "" if op0_is_default else \
            f"{PART_SEPARATOR}{op0}"

    def __str__(self) -> str:
        """
        Get the suffix for the name of the algorithm by subclasses.

        :return: the name of the algorithm
        """
        return self._name_suffix

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

    def __init__(self,
                 op0: Op0,
                 op1: Op1,
                 op0_is_default: bool = True,
                 op1_is_default: bool = False) -> None:
        """
        Create the algorithm with nullary and unary search operator.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param op0_is_default: is this a default nullary operator?
            If `True`, it will not be included in the name suffix.
        :param op1_is_default: is this a default unary operator?
            If `True`, it will not be included in the name suffix.
        """
        super().__init__(op0=op0,
                         op0_is_default=op0_is_default)
        #: The unary search operator.
        self.op1: Final[Op1] = check_op1(op1)
        if not isinstance(op1_is_default, bool):
            raise type_error(op1_is_default, "op1_is_default", bool)
        #: the internal name suffix
        if not op1_is_default:
            self._name_suffix += f"{PART_SEPARATOR}{op1}"

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

    def __init__(self,
                 op0: Op0,
                 op1: Op1,
                 op2: Op2,
                 op0_is_default: bool = True,
                 op1_is_default: bool = False,
                 op2_is_default: bool = False) -> None:
        """
        Create the algorithm with nullary and unary search operator.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param op2: the binary search operator
        :param op0_is_default: is this a default nullary operator?
            If `True`, it will not be included in the name suffix.
        :param op1_is_default: is this a default unary operator?
            If `True`, it will not be included in the name suffix.
        :param op2_is_default: is this a default binary operator?
            If `True`, it will not be included in the name suffix.
        """
        super().__init__(op0=op0,
                         op1=op1,
                         op0_is_default=op0_is_default,
                         op1_is_default=op1_is_default)
        #: The binary search operator.
        self.op2: Final[Op2] = check_op2(op2)
        if not isinstance(op2_is_default, bool):
            raise type_error(op2_is_default, "op2_is_default", bool)
        #: the internal name suffix
        if not op2_is_default:
            self._name_suffix += f"{PART_SEPARATOR}{op2}"

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_OP2) as sc:
            self.op2.log_parameters_to(sc)


class CallableAlgorithm(CallableComponent, Algorithm):
    """Wrapping a Callable such as a lambda into an algorithm."""

    def __init__(self,
                 algorithm: Callable[[Process], None],
                 name: str = None) -> None:
        """
        Create a wrapper mapping a Callable to an optimization algorithm.

        :param algorithm: the algorithm implementation to wrap, can be a
            lambda expression.
        :param name: the name of the algorithm
        """
        super().__init__(inner=algorithm,
                         name="unnamed_algorithm" if (name is None) else name)
        self.solve = algorithm  # type: ignore


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
