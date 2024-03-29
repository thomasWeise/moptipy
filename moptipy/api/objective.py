"""
The base class for implementing objective functions.

An objective function evaluates the quality of a candidate solution of an
optimization problem. Solutions with smaller objective values are better,
i.e., objective functions are subject to minimization. All objective
functions inherit from :class:`~moptipy.api.objective.Objective`. If you
implement a new objective function, you can test it via the pre-defined unit
test routine :func:`~moptipy.tests.objective.validate_objective`.
"""
from math import inf, isfinite
from typing import Any

from pycommons.types import type_error

from moptipy.api import logging
from moptipy.api.component import Component
from moptipy.utils.logger import KeyValueLogSection


# start book
class Objective(Component):
    """
    An objective function subject to minimization.

    An objective function represents one optimization criterion that
    is used for rating the solution quality. All objective functions in
    our system are subject to minimization, meaning that smaller values
    are better.
    """

    def evaluate(self, x) -> float | int:
        """
        Evaluate a solution `x` and return its objective value.

        The return value is either an integer or a float and must be
        finite. Smaller objective values are better, i.e., all objective
        functions are subject to minimization.

        :param x: the candidate solution
        :return: the objective value
        """
    # end book

    def lower_bound(self) -> float | int:
        """
        Get the lower bound of the objective value.

        This function returns a theoretical limit for how good a solution
        could be at best. If no real limit is known, the function should
        return `-inf`.

        :return: the lower bound of the objective value
        """
        return -inf

    def upper_bound(self) -> float | int:
        """
        Get the upper bound of the objective value.

        This function returns a theoretical limit for how bad a solution could
        be at worst. If no real limit is known, the function should return
        `inf`.

        :return: the upper bound of the objective value
        """
        return inf

    def is_always_integer(self) -> bool:
        """
        Return `True` if :meth:`~evaluate` will always return an `int` value.

        :returns: `True` if :meth:`~evaluate` will always return an `int`
          or `False` if also a `float` may be returned.
        """
        return False

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this function to the provided destination.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        b = self.lower_bound()
        if isinstance(b, int) or isfinite(b):
            logger.key_value(logging.KEY_F_LOWER_BOUND, b)
        b = self.upper_bound()
        if isinstance(b, int) or isfinite(b):
            logger.key_value(logging.KEY_F_UPPER_BOUND, b)


def check_objective(objective: Any) -> Objective:
    """
    Check whether an object is a valid instance of :class:`Objective`.

    :param objective: the objective
    :return: the objective
    :raises TypeError: if `objective` is not an instance of
        :class:`Objective`

    >>> check_objective(Objective())
    Objective
    >>> try:
    ...     check_objective('A')
    ... except TypeError as te:
    ...     print(te)
    objective function should be an instance of moptipy.api.objective.\
Objective but is str, namely 'A'.
    >>> try:
    ...     check_objective(None)
    ... except TypeError as te:
    ...     print(te)
    objective function should be an instance of moptipy.api.objective.\
Objective but is None.
    """
    if isinstance(objective, Objective):
        return objective
    raise type_error(objective, "objective function", Objective)
