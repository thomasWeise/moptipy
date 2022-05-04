"""This module provides the class for implementing objective functions."""
from math import inf
from typing import Union, Callable, Optional, Final, Any

from moptipy.api import logging
from moptipy.api.component import CallableComponent, Component
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error


# start book
class Objective(Component):
    """
    An objective function subject to minimization.

    An objective function represents one optimization criterion that
    is used for rating the solution quality. All objective functions in
    our system are subject to minimization, meaning that smaller values
    are better.
    """

    def evaluate(self, x) -> Union[float, int]:
        """
        Evaluate a solution `x` and return its objective value.

        The return value is either an integer or a float and must be
        finite. Smaller objective values are better, i.e., all objective
        functions are subject to minimization.

        :param x: the candidate solution
        :return: the objective value
        """
    # end book

    def lower_bound(self) -> Union[float, int]:
        """
        Get the lower bound of the objective value.

        This function returns a theoretical limit for how good a solution
        could be at best. If no real limit is known, the function should
        return `-inf`.

        :return: the lower bound of the objective value
        """
        return -inf

    def upper_bound(self) -> Union[float, int]:
        """
        Get the upper bound of the objective value.

        This function returns a theoretical limit for how bad a solution could
        be at worst. If no real limit is known, the function should return
        `inf`.

        :return: the upper bound of the objective value
        """
        return inf


class CallableObjective(CallableComponent, Objective):
    """Wrapping a Callable such as a lambda into an objective function."""

    def __init__(self,
                 function: Callable[[Any], Union[float, int]],
                 lower_bound: Union[float, int] = -inf,
                 upper_bound: Union[float, int] = inf,
                 name: Optional[str] = None) -> None:
        """
        Create a wrapper mapping a Callable to an objective function.

        :param function: the function to wrap, can be a lambda expression
        :param lower_bound: the lower bound of the objective function
        :param upper_bound: the upper bound of the objective function
        :param name: the name of the objective function
        """
        super().__init__(inner=function,
                         name="unnamed_function" if (name is None) else name)

        if not (isinstance(lower_bound, (int, float))):
            raise type_error(lower_bound, "lower_bound", (int, float))

        if not (isinstance(upper_bound, (int, float))):
            raise type_error(upper_bound, "upper_bound", (int, float))

        if lower_bound >= upper_bound:
            raise ValueError(f"lower_bound {lower_bound} "
                             "must be less than upper_bound "
                             f"{upper_bound} but is not.")

        self.__lower_bound: Final[Union[int, float]] = lower_bound
        self.__upper_bound: Final[Union[int, float]] = upper_bound
        self.evaluate = self._inner  # type: ignore

    def lower_bound(self) -> Union[float, int]:
        """Get the lower bound of the objective value."""
        return self.__lower_bound

    def upper_bound(self) -> Union[float, int]:
        """Get the upper bound of the objective value."""
        return self.__upper_bound

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this function to the provided destination.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_F_LOWER_BOUND, self.__lower_bound)
        logger.key_value(logging.KEY_F_UPPER_BOUND, self.__upper_bound)


def check_objective(objective: Objective) -> Objective:
    """
    Check whether an object is a valid instance of :class:`Objective`.

    :param objective: the objective
    :return: the object
    :raises TypeError: if `objective` is not an instance of
        :class:`Objective`
    """
    if not isinstance(objective, Objective):
        raise type_error(objective, "objective function", Objective)
    return objective
