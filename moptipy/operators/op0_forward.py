"""A nullary operator forwarding to another function."""
from typing import Any, Callable

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op0
from moptipy.utils.types import type_error


class Op0Forward(Op0):
    """Get the forwarder function."""

    def __init__(self):
        """Initialize this operator."""
        #: the internal blueprint for filling permutations
        self.__call: Callable[[Any], None] | None = None

    def op0(self, random: Generator, dest: np.ndarray) -> None:
        """Forward the call."""
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
