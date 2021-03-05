"""The base classes for implementing optimization algorithms."""
from abc import ABC, abstractmethod
from typing import Callable

from moptipy.api.component import Component
from moptipy.api.objective import _CallableComponent
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.api.process import Process
from moptipy.utils import logging
from moptipy.utils.logger import KeyValueSection


class Algorithm(Component):
    """A class to represent an optimization algorithm."""

    @abstractmethod
    def solve(self, process: Process) -> None:
        """
        Apply this optimization algorithm to the given black-box process.

        :param moptipy.api.Process process: the black-box process
        """
        raise NotImplementedError


class Algorithm0(Algorithm, ABC):
    """An algorithm implementation with a nullary search operator."""

    def __init__(self,
                 op0: Op0,
                 op0_is_default: bool = True) -> None:
        """
        Create the algorithm with nullary search operator.

        :param Op0 op0: the nullary search operator
        :param bool op0_is_default: is this a default nullary operator?
            if `True`, it will not be included in the name suffix
        """
        if (op0 is None) or (not isinstance(op0, Op0)):
            TypeError("op0 must be instance of Op0, but is '"
                      + str(type(op0)) + "'")
        self.op0 = op0
        if not isinstance(op0_is_default, bool):
            raise TypeError("op0_is_default must be bool, but is "
                            + str(type(op0_is_default)) + ".")
        self._op0_is_default = op0_is_default

    def get_name(self) -> str:
        """
        Get the suffix for the name of the algorithm by subclasses.

        :return: the name of the algorithm
        :rtype: str
        """
        return "" if self._op0_is_default else self.op0.get_name()

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param KeyValueLogger logger: the logger
        """
        super().log_parameters_to(logger)
        with logger.scope(logging.SCOPE_OP0) as sc:
            self.op0.log_parameters_to(sc)


class Algorithm1(Algorithm0, ABC):
    """An algorithm implementation with a unary search operator."""

    def __init__(self,
                 op0: Op0,
                 op1: Op1,
                 op0_is_default: bool = True,
                 op1_is_default: bool = False) -> None:
        """
        Create the algorithm with nullary and unary search operator.

        :param Op0 op0: the nullary search operator
        :param Op1 op1: the unary search operator
        :param bool op0_is_default: is this a default nullary operator?
            If `True`, it will not be included in the name suffix.
        :param bool op1_is_default: is this a default unary operator?
            If `True`, it will not be included in the name suffix.
        """
        super().__init__(op0=op0,
                         op0_is_default=op0_is_default)
        if (op1 is None) or (not isinstance(op1, Op1)):
            TypeError("op1 must be instance of Op1, but is '"
                      + str(type(op1)) + "'")
        self.op1 = op1
        if not isinstance(op1_is_default, bool):
            raise TypeError("op1_is_default must be bool, but is "
                            + str(type(op1_is_default)) + ".")
        self._op1_is_default = op1_is_default

    def get_name(self) -> str:
        """
        Get the suffix for the name of the algorithm by subclasses.

        :return: the name of the algorithm
        :rtype: str
        """
        return logging.sanitize_names([
            "" if self._op0_is_default else self.op0.get_name(),
            "" if self._op1_is_default else self.op1.get_name()])

    def log_parameters_to(self, logger: KeyValueSection):
        """
        Log the parameters of the algorithm to a logger.

        :param KeyValueLogger logger: the logger
        """
        super().log_parameters_to(logger)
        with logger.scope(logging.SCOPE_OP1) as sc:
            self.op1.log_parameters_to(sc)


class Algorithm2(Algorithm1, ABC):
    """An algorithm implementation with a binary and unary operator."""

    def __init__(self,
                 op0: Op0,
                 op1: Op1,
                 op2: Op2,
                 op0_is_default: bool = True,
                 op1_is_default: bool = False,
                 op2_is_default: bool = False) -> None:
        """
        Create the algorithm with nullary and unary search operator.

        :param Op0 op0: the nullary search operator
        :param Op1 op1: the unary search operator
        :param Op2 op2: the binary search operator
        :param bool op0_is_default: is this a default nullary operator?
            If `True`, it will not be included in the name suffix.
        :param bool op1_is_default: is this a default unary operator?
            If `True`, it will not be included in the name suffix.
        :param bool op2_is_default: is this a default binary operator?
            If `True`, it will not be included in the name suffix.
        """
        super().__init__(op0=op0,
                         op1=op1,
                         op0_is_default=op0_is_default,
                         op1_is_default=op1_is_default)
        if (op2 is None) or (not isinstance(op2, Op2)):
            TypeError("op2 must be instance of Op2, but is '"
                      + str(type(op2)) + "'")
        self.op2 = op2
        if not isinstance(op2_is_default, bool):
            raise TypeError("op2_is_default must be bool, but is "
                            + str(type(op2_is_default)) + ".")
        self._op2_is_default = op2_is_default

    def get_name(self) -> str:
        """
        Get the suffix for the name of the algorithm by subclasses.

        :return: the name of the algorithm
        :rtype: str
        """
        return logging.sanitize_names([
            "" if self._op0_is_default else self.op0.get_name(),
            "" if self._op1_is_default else self.op1.get_name(),
            "" if self._op2_is_default else self.op2.get_name()])

    def log_parameters_to(self, logger: KeyValueSection):
        """
        Log the parameters of the algorithm to a logger.

        :param KeyValueLogger logger: the logger
        """
        super().log_parameters_to(logger)
        with logger.scope(logging.SCOPE_OP2) as sc:
            self.op2.log_parameters_to(sc)


class CallableAlgorithm(_CallableComponent, Algorithm):
    """Wrapping a Callable such as a lambda into an algorithm."""

    def __init__(self,
                 algorithm: Callable,
                 name: str = None) -> None:
        """
        Create a wrapper mapping a Callable to an optimization algorithm.

        :param Callable algorithm: the algorithm implementation to wrap,
            can be a lambda expression.
        :param str name: the name of the algorithm
        """
        super().__init__(inner=algorithm,
                         name="unnamed_algorithm" if (name is None) else name)

    def solve(self, process: Process) -> None:
        """
        Apply the inner callable to the search process.

        :param moptipy.api.Process process: the search process
        """
        self._inner(process)


def _check_algorithm(algorithm: Algorithm) -> Algorithm:
    """
    Check whether an object is a valid instance of :class:`Algorithm`.

    :param moptipy.api.Algorithm algorithm: the algorithm object
    :return: the object
    :raises TypeError: if `algorithm` is not an instance of :class:`Algorithm`
    """
    if algorithm is None:
        raise TypeError("An algorithm must not be None.")
    if not isinstance(algorithm, Algorithm):
        raise TypeError(
            "An algorithm must be instance of Algorithm, but is "
            + str(type(algorithm)) + ".")
    return algorithm
