"""
A nullary operator forwarding to another function.

This is a nullary operator (an instance of
:class:`~moptipy.api.operators.Op0`) whose method
:meth:`~moptipy.api.operators.Op0.op0` forwards to another `Callable`.
This other `Callable` can then return a solution that is created in some
special way, or maybe even the current best solution of a search process.

This operator has been designed to be used in conjunction with
:func:`~moptipy.api.subprocesses.from_starting_point`, which is an
optimization :class:`~moptipy.api.process.Process` where a starting point
has been defined, i.e., where the methods
:meth:`~moptipy.api.process.Process.get_copy_of_best_x` and
:meth:`~moptipy.api.process.Process.get_best_f` return pre-defined values.
By setting :meth:`~moptipy.operators.op0_forward.Op0Forward.forward_to` to
:meth:`~moptipy.api.process.Process.get_copy_of_best_x`, this nullary operator
will return the current-best solution of the optimization process, which, in
this case, will be the pre-defined starting point.
Any optimization algorithm (e.g., an instance of
:class:`~moptipy.api.algorithm.Algorithm0`) using this nullary operator to get
its initial solution will then begin the search at this pre-defined starting
point. This allows using one algorithm as a sub-algorithm of another one.
Wrapping :func:`~moptipy.api.subprocesses.from_starting_point` around the
result of a call to :func:`~moptipy.api.subprocesses.for_fes` would allow to
limit the number of objective function evaluations consumed by the
sub-algorithm.
"""
from typing import Any, Callable

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op0
from moptipy.utils.types import type_error


class Op0Forward(Op0):
    """A nullary operator that forwards all calls to `op0` to a `Callable`."""

    def __init__(self):
        """Initialize this operator."""
        #: the internal blueprint for filling permutations
        self.__call: Callable[[Any], None] | None = None

    def op0(self, random: Generator, dest: np.ndarray) -> None:
        """
        Forward the call.

        :param random: ignored
        :param dest: the destination data structure to be filled with the data
            of the point in the search space by the internal `Callable` set by
            :meth:`forward_to`.
        """
        self.__call(dest)

    def forward_to(self, call: Callable[[Any], None]) -> None:
        """
        Set the `Callable` to forward all calls from :meth:`op0` to.

        :param call: the `Callable` to which all calls to :meth:`op0` should
            be delegated to.
        """
        if not callable(call):
            raise type_error(call, "call", call=True)
        self.__call = call

    def stop_forwarding(self) -> None:
        """Stop forwarding the call."""
        self.__call = None

    def __str__(self) -> str:
        """
        Get the name of this operator.

        :return: "forward"
        """
        return "forward"

    def initialize(self) -> None:
        """Initialize this operator by stopping to forward."""
        super().initialize()
        self.stop_forwarding()
