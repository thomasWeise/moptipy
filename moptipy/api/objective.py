"""This module provides the class for implementing objective functions."""
from abc import abstractmethod
from math import inf
from typing import Union, Callable, Optional, Final

from moptipy.api.component import _CallableComponent, Component
from moptipy.utils import logging
from moptipy.utils.logger import KeyValueSection


class Objective(Component):
    """An objective function: a criterion rating the solution quality."""

    @abstractmethod
    def evaluate(self, x) -> Union[float, int]:
        """
        Evaluate a solution `x` and return its objective value.

        The return value is either an integer or a float and must be finite.
        Smaller objective values are better, i.e., all objective functions are
        subject to minimization.

        :param x: the candidate solution
        :return: the objective value
        :rtype: Union[float, int]
        """
        raise NotImplementedError

    def lower_bound(self) -> Union[float, int]:
        """
        Get the lower bound of the objective value.

        This function returns a theoretically limit for how good a
        solution could be at best. If no real limit is known, the
        function should return `-inf`.

        :return: the lower bound of the objective value
        :rtype: Union[float, int]
        """
        return -inf

    def upper_bound(self) -> Union[float, int]:
        """
        The upper bound of the objective value.

        This function returns a theoretical limit for how bad a solution could
        be at worst. If no real limit is known, the function should return
        `inf`.

        :return: the upper bound of the objective value
        :rtype: Union[float, int]
        """
        return inf


class CallableObjective(_CallableComponent, Objective):
    """Wrapping a Callable such as a lambda into an objective function."""

    def __init__(self,
                 function: Callable,
                 lower_bound: Union[float, int] = -inf,
                 upper_bound: Union[float, int] = inf,
                 name: Optional[str] = None) -> None:
        """
        Create a wrapper mapping a Callable to an objective function.

        :param Callable function: the function to wrap,
            can be a lambda expression
        :param Union[float, int] lower_bound:
            the lower bound of the objective function
        :param Union[float, int] upper_bound:
            the upper bound of the objective function
        :param Optional[str] name: the name of the objective function
        """
        super().__init__(inner=function,
                         name="unnamed_function" if (name is None) else name)

        if not (isinstance(lower_bound, (int, float))):
            raise TypeError("lower_bound must be either int or float, "
                            f"but is {type(lower_bound)}.")

        if not (isinstance(upper_bound, (int, float))):
            raise TypeError("upper_bound must be either int or float, "
                            f"but is {type(upper_bound)}.")

        if lower_bound >= upper_bound:
            raise ValueError(f"lower_bound {lower_bound} "
                             "must be less than upper_bound "
                             f"{upper_bound} but is not.")

        self.__lower_bound: Final[Union[int, float]] = lower_bound
        self.__upper_bound: Final[Union[int, float]] = upper_bound

    def evaluate(self, x) -> Union[float, int]:
        """
        Invoke the internal callable to get the objective value.

        :param x: the candidate solution
        :return: the objective value
        :rtype: Union[float, int]
        """
        return self._inner(x)

    def lower_bound(self) -> Union[float, int]:
        """
        Provide the lower bound passed into the constructor.

        :return: the lower bound of the objective value
        """
        return self.__lower_bound

    def upper_bound(self) -> Union[float, int]:
        """
        Provide the upper bound passed into the constructor.

        :return: the upper bound of the objective value
        """
        return self.__upper_bound

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters of this function to the provided destination.

        :param moptipy.utils.KeyValueSection logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_F_LOWER_BOUND, self.__lower_bound)
        logger.key_value(logging.KEY_F_UPPER_BOUND, self.__upper_bound)


def _check_objective(objective: Objective) -> Objective:
    """
    Check whether an object is a valid instance of :class:`Objective`.

    :param Objective objective: the object
    :return: the object
    :raises TypeError: if `objective` is not an instance of
    :class:`Objective`
    """
    if objective is None:
        raise TypeError("An objective function must not be None.")
    if not isinstance(objective, Objective):
        raise TypeError("An objective function must be instance of "
                        f"Objective, but is {type(objective)}.")
    return objective
