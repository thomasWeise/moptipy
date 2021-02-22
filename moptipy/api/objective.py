from abc import abstractmethod
from math import inf
from typing import Union, Callable

from ..utils.logger import KeyValues
from ..utils import logging
from .component import _CallableComponent, Component


class Objective(Component):
    """
    A class to represent an objective function, i.e., an criterion rating the quality of a solution.
    """

    @abstractmethod
    def evaluate(self, x) -> Union[float, int]:
        """
        Evaluate a candidate solution x and return its objective value as either integer or float.

        Smaller objective values are better, i.e., all objective functions are subject to minimization.

        :param x: the candidate solution
        :return: the objective value
        :rtype: Union[float, int]
        """
        raise NotImplementedError

    def lower_bound(self) -> Union[float, int]:
        """
        The lower bound of the objective value - a theoretically limit for how good a solution could be at best

        :return: the lower bound of the objective value
        :rtype: Union[float, int]
        """
        return -inf

    def upper_bound(self) -> Union[float, int]:
        """
        The upper bound of the objective value - a theoretical limit for how bad a solution could be at worst

        :return: the upper bound of the objective value
        :rtype: Union[float, int]
        """
        return inf


class CallableObjective(_CallableComponent, Objective):
    """An class for wrapping a Callable such as a lambda into an objective function."""

    def __init__(self,
                 function: Callable,
                 lower_bound: Union[float, int] = -inf,
                 upper_bound: Union[float, int] = inf,
                 name: str = None):
        """
        Create a wrapper mapping a Callable to an objective function

        :param Callable function: the function to wrap, can be a lambda expression
        :param Union[float, int] lower_bound: the lower bound of the objective function
        :param Union[float, int] upper_bound: the upper bound of the objective function
        :param str name: the name of the objective function
        """
        super().__init__(inner=function,
                         name="unnamed_function" if (name is None) else name)

        if not (isinstance(lower_bound, int) or isinstance(lower_bound, float)):
            raise ValueError("lower_bound must be either int or float, but is "
                             + str(type(lower_bound)))

        if not (isinstance(upper_bound, int) or isinstance(upper_bound, float)):
            raise ValueError("upper_bound must be either int or float, but is "
                             + str(type(upper_bound)))

        if lower_bound >= upper_bound:
            raise ValueError("lower_bound " + str(lower_bound) +
                             "must be less than upper_bound "
                             + str(upper_bound) + " but is not.")

        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound

    def evaluate(self, x) -> Union[float, int]:
        return self._inner(x)

    def lower_bound(self) -> Union[float, int]:
        return self.__lower_bound

    def upper_bound(self) -> Union[float, int]:
        return self.__upper_bound

    def log_parameters_to(self, logger: KeyValues):
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_F_LOWER_BOUND, self.__lower_bound)
        logger.key_value(logging.KEY_F_UPPER_BOUND, self.__upper_bound)
