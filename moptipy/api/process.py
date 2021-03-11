"""
Processes offer data to both the user and the optimization algorithm.

They provide the information about the optimization process and its current
state as handed to the optimization algorithm and, after the algorithm has
finished, to the user.
"""
from abc import abstractmethod
from typing import Optional, Union

from numpy.random import Generator

from moptipy.api.objective import Objective
from moptipy.api.space import Space


class Process(Space, Objective):
    """
    Processes offer data to both the user and the optimization algorithm.

    A :class:`Process` provides an optimization algorithm access to a problem
    as well as information about the best-so-far results and how much
    runtime was consumed.
    It also lets the user access the final result of optimization and
    can be implemented to write log files.
    """

    def __init__(self) -> None:
        """Internal method to initialize the process. Do not call directly."""
        #: This will be `True` after :meth:`terminate` has been called.
        self._terminated: bool = False
        #: This becomes `True` when :meth:`should_terminate` returned `True`.
        self._knows_that_terminated: bool = False

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
        Check whether the optimization process should terminate.

        If this function returns `True`, the optimization process must
        not perform any objective function evaluations anymore.
        It will automatically become `True` when a termination criterion
        is hit or if anyone calls :meth:`terminate`, which happens also
        at the end of a `with` statement.

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
        Obtain the number consumed objective function evaluations.

        This is the number of calls to :meth:`evaluate`.

        :return: the number of objective function evaluations so far
        :rtype: int
        """
        raise NotImplementedError

    @abstractmethod
    def get_consumed_time_millis(self) -> int:
        """
        Obtain an approximation of the consumed runtime in milliseconds.

        :return: the consumed runtime measured in milliseconds.
        :rtype: int
        """
        raise NotImplementedError

    @abstractmethod
    def get_max_fes(self) -> Optional[int]:
        """
        Obtain the maximum number of permitted objective function evaluations.

        If no limit is set, `None` is returned.

        :return: the maximum number of objective function evaluations,
            or `None` if no limit is specified.
        :rtype: Optional[int]
        """
        raise NotImplementedError

    @abstractmethod
    def get_max_time_millis(self) -> Optional[int]:
        """
        Obtain the maximum runtime permitted in milliseconds.

        If no limit is set, `None` is returned.

        :return: the maximum runtime permitted in milliseconds,
            or `None` if no limit is specified.
        :rtype: Optional[int]
        """
        raise NotImplementedError

    @abstractmethod
    def has_current_best(self) -> bool:
        """
        Check whethers a current best solution is available.

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
    def get_copy_of_current_best_x(self, x) -> None:
        """
        Get a copy of the current best point in the search space.

        :param x: the destination data structure to be overwritten
        """
        raise NotImplementedError

    @abstractmethod
    def get_copy_of_current_best_y(self, y) -> None:
        """
        Get a copy of the current best point in the solution space.

        :param y: the destination data structure to be overwritten
        """
        raise NotImplementedError

    @abstractmethod
    def get_last_improvement_fe(self) -> int:
        """
        Get the FE at which the last improvement was made.

        :return: the function evaluation when the last improvement was made
        :rtype: int
        :raises ValueError: if no FE was performed yet
        """
        raise NotImplementedError

    @abstractmethod
    def get_last_improvement_time_millis(self) -> int:
        """
        Get the FE at which the last improvement was made.

        :return: the function evaluation when the last improvement was made
        :rtype: int
        :raises ValueError: if no FE was performed yet
        """
        raise NotImplementedError

    def get_name(self) -> str:
        """
        Get the name of this process implementation.

        :return: "process"
        """
        return "process"

    def terminate(self) -> None:
        """
        Terminate this process.

        This function is automatically called at the end of the `with`
        statement, but can also be called by the algorithm when it is
        finished and is also invoked automatically when a termination
        criterion is hit.
        After the first time this method is invoked, :meth:should_terminate`
        becomes `True`.
        """
        self._terminated = True

    def __enter__(self) -> 'Process':
        """
        Begin a `with` statement.

        :return: this process itself
        :rtype: Process
        """
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        End a `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        """
        self.terminate()
