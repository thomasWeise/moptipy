from abc import abstractmethod
from typing import Optional, Union

from numpy.random import Generator

from .objective import Objective
from .space import Space


class Process(Space, Objective):
    """
    A black-box optimization process provides an optimization algorithm access to a problem instance.
    """

    def __init__(self):
        self._terminated = False
        self._knows_that_terminated = False

    @abstractmethod
    def get_random(self) -> Generator:
        """
        Obtain the random number generator.

        :return: the random number generator
        :rtype: Generator
        """
        raise NotImplementedError

    def should_terminate(self) -> bool:
        """
        Should the optimization process terminate?

        If this function returns false, the optimization process must
        not perform any objective function evaluations anymore.

        :return: True if the process should terminate, False if not
        :rtype: Generator
        """
        if self._terminated:
            self._knows_that_terminated = True
            return True
        return False

    @abstractmethod
    def get_consumed_fes(self) -> int:
        """
        Obtain the number consumed objective function evaluations, i.e., the number of calls to evaluate(...)

        :return: the number of objective function evaluations so far
        :rtype: int
        """
        raise NotImplementedError

    @abstractmethod
    def get_consumed_time_millis(self) -> int:
        """
        Obtain the consumed runtime measured in milliseconds.

        :return: the consumed runtime measured in milliseconds.
        :rtype: int
        """
        raise NotImplementedError

    @abstractmethod
    def get_max_fes(self) -> Optional[int]:
        """
        Obtain the maximum number of objective function evaluations, or None if no limit is specified.

        :return: the maximum number of objective function evaluations, or None if no limit is specified.
        :rtype: Optional[int]
        """
        raise NotImplementedError

    @abstractmethod
    def get_max_time_millis(self) -> Optional[int]:
        """
        Obtain the maximum runtime permitted in milliseconds, or None if no limit is specified.

        :return: the maximum runtime permitted in milliseconds, or None if no limit is specified.
        :rtype: Optional[int]
        """
        raise NotImplementedError

    @abstractmethod
    def has_current_best(self) -> bool:
        """
        Is a current best solution available?

        As soon as one objective function evaluation has been performed,
        the black-box process can provide a best-so-far solution. Then,
        this method returns True. Otherwise, it returns False.

        :return: True if the current-best solution can be queried.
        :rtype: bool
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_best_f(self) -> Union[int, float]:
        """
        Get the objective value of the current best solution.

        :return: the objective value of the current best solution.
        :rtype: Union[int,float]
        """
        raise NotImplementedError

    @abstractmethod
    def get_copy_of_current_best_x(self, x):
        """
        Get a copy of the current best point in the search space.

        :param x: the destination data structure to be overwritten by that point
        """
        raise NotImplementedError

    @abstractmethod
    def get_copy_of_current_best_y(self, y):
        """
        Get a copy of the current best point in the solution space.

        :param y: the destination data structure to be overwritten by that point
        """
        raise NotImplementedError

    @abstractmethod
    def get_last_improvement_fe(self) -> int:
        """ The the function evaluation at which the last improvement was made. """
        raise NotImplementedError

    @abstractmethod
    def get_last_improvement_time_millis(self) -> int:
        """ The the consumed milliseconds since the start at which the last improvement was made. """
        raise NotImplementedError

    @abstractmethod
    def log_state(self, key: str, value: Union[bool, int, float]):
        """
        Log a dynamic state value for a specific key.

        :param str key: the key which should be logged
        :param Union[bool, int, float] value: the value to be logged
        """
        pass

    def get_name(self):
        return "Process"

    def terminate(self):
        """ Terminate this process. """
        self._terminated = True

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.terminate()
