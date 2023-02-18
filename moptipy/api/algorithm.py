"""
The base classes for implementing optimization algorithms.

All optimization algorithms implemented based on the `moptipy` API inherit
from :class:`~moptipy.api.algorithm.Algorithm`. If you implement a new
algorithm, you will want to override the following methods:

1. :meth:`~moptipy.api.algorithm.Algorithm.solve` implements the algorithm
   itself. It receives an instance of :class:`~moptipy.api.process.Process` as
   parameter that allows for the creation and evaluation of candidate
   solutions and that provides a random number generator. The optimization
   algorithm then will sample solutions and pass them to
   :meth:`~moptipy.api.process.Process.evaluate` to obtain their objective
   value, striving sampling better and better solutions.
2. The dunder method `__str__` should be overridden to return a short mnemonic
   name of the algorithm.
3. :meth:`~moptipy.api.component.Component.log_parameters_to` needs to be
   overridden if the algorithm has any parameters. This methods then should
   store the values of all the parameters to the logging context. It should
   also invoke the :meth:`~moptipy.api.component.Component.log_parameters_to`
   routines of all sub-components of the algorithm.
4. :meth:`~moptipy.api.component.Component.initialize` needs to be overridden
   to reset/initialize all internal data structures and to invoke all the
   :meth:`~moptipy.api.component.Component.initialize` of all components (such
   as search operators) of the algorithm.

Notice that we already provide specialized algorithm sub-classes for several
common scenarios, such as:

1. :class:`~moptipy.api.algorithm.Algorithm0` for algorithms that have a
   nullary search operator (:class:`~moptipy.api.operators.Op0`).
2. :class:`~moptipy.api.algorithm.Algorithm1` for algorithms that have  a
   nullary (:class:`~moptipy.api.operators.Op0`) and an unary
   (:class:`~moptipy.api.operators.Op1`) search operator.
3. :class:`~moptipy.api.algorithm.Algorithm2` for algorithms that have  a
   nullary (:class:`~moptipy.api.operators.Op0`), an unary
   (:class:`~moptipy.api.operators.Op1`), and a binary
   (:class:`~moptipy.api.operators.Op2`) search operator.
4. :class:`~moptipy.api.mo_algorithm.MOAlgorithm` for multi-objective
   optimization problems.

These classes automatically invoke the
:meth:`~moptipy.api.component.Component.log_parameters_to` and
:meth:`~moptipy.api.component.Component.initialize` routines of their
operators.

If you implement a new algorithm, you can and should test with the pre-defined
unit test routine :func:`~moptipy.tests.algorithm.validate_algorithm`, or its
specialized versions

1. for bit-string based search spaces based on
   :func:`~moptipy.tests.on_bitstrings.validate_algorithm_on_bitstrings`):
   a. :func:`~moptipy.tests.on_bitstrings.validate_algorithm_on_onemax`,
   b. :func:`~moptipy.tests.on_bitstrings.validate_algorithm_on_leadingones`
2. for the JSSP based on
   :func:`~moptipy.tests.on_jssp.validate_algorithm_on_1_jssp`:
   a. :func:`~moptipy.tests.on_jssp.validate_algorithm_on_jssp`
3. on real-valued vector search spaces based on
   :func:`~moptipy.tests.on_vectors.validate_algorithm_on_vectors`):
   a. :func:`~moptipy.tests.on_vectors.validate_algorithm_on_ackley`
"""
from typing import Any, Final

from moptipy.api.component import Component
from moptipy.api.logging import SCOPE_OP0, SCOPE_OP1, SCOPE_OP2
from moptipy.api.operators import (
    Op0,
    Op1,
    Op2,
    check_op0,
    check_op1,
    check_op2,
)
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
            raise ValueError(f"Algorithm name cannot be {name!r}.")
        #: the name of this optimization algorithm, which is also the return
        #: value of `__str__()`
        self.name: Final[str] = name

    def __str__(self) -> str:
        """
        Get the name of the algorithm.

        :return: the name of the algorithm
        """
        return self.name

    def initialize(self) -> None:
        """Initialize the algorithm."""
        super().initialize()
        self.op0.initialize()

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

    def initialize(self) -> None:
        """Initialize the algorithm."""
        super().initialize()
        self.op1.initialize()

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
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

    def initialize(self) -> None:
        """Initialize the algorithm."""
        super().initialize()
        self.op2.initialize()

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_OP2) as sc:
            self.op2.log_parameters_to(sc)


def check_algorithm(algorithm: Any) -> Algorithm:
    """
    Check whether an object is a valid instance of :class:`Algorithm`.

    :param algorithm: the algorithm object
    :return: the object
    :raises TypeError: if `algorithm` is not an instance of :class:`Algorithm`

    >>> check_algorithm(Algorithm())
    Algorithm
    >>> try:
    ...     check_algorithm('A')
    ... except TypeError as te:
    ...     print(te)
    algorithm should be an instance of moptipy.api.algorithm.\
Algorithm but is str, namely 'A'.
    >>> try:
    ...     check_algorithm(None)
    ... except TypeError as te:
    ...     print(te)
    algorithm should be an instance of moptipy.api.algorithm.\
Algorithm but is None.
    """
    if isinstance(algorithm, Algorithm):
        return algorithm
    raise type_error(algorithm, "algorithm", Algorithm)
