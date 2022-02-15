"""
Processes offer data to both the user and the optimization algorithm.

They provide the information about the optimization process and its current
state as handed to the optimization algorithm and, after the algorithm has
finished, to the user.
"""
from math import inf, isnan
from typing import Optional, Union, ContextManager

from numpy.random import Generator

from moptipy.api.objective import Objective
from moptipy.api.space import Space


# start book
class Process(Space, Objective, ContextManager):
    """
    Processes offer data to the optimization algorithm and the user.

    A Process presents the objective function and search space to an
    optimization algorithm. Since it wraps the actual objective
    function, it can see all evaluated solutions and remember the
    best-so-far solution. It can also count the FEs and the runtime
    that has passed. Therefore, it also presents the termination
    criterion to the optimization algorithm. It also provides a random
    number generator the algorithm. It can a write log files with the
    progress of the search and the end result. Finally, it provides
    the end result to the user, who can access it after the algorithm
    has finished.
    """

# end book

    def __init__(self) -> None:
        """Initialize the process. Do not call directly."""
        #: This will be `True` after :meth:`terminate` has been called.
        self._terminated: bool = False
        #: This becomes `True` when :meth:`should_terminate` returned `True`.
        self._knows_that_terminated: bool = False

    def get_random(self) -> Generator:  # +book
        """
        Obtain the random number generator.

        :return: the random number generator
        :rtype: Generator
        """

    def should_terminate(self) -> bool:  # +book
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
        return False  # +book

    def get_consumed_fes(self) -> int:
        """
        Obtain the number consumed objective function evaluations.

        This is the number of calls to :meth:`evaluate`.

        :return: the number of objective function evaluations so far
        :rtype: int
        """

    def get_consumed_time_millis(self) -> int:
        """
        Obtain an approximation of the consumed runtime in milliseconds.

        :return: the consumed runtime measured in milliseconds.
        :rtype: int
        """

    def get_max_fes(self) -> Optional[int]:
        """
        Obtain the maximum number of permitted objective function evaluations.

        If no limit is set, `None` is returned.

        :return: the maximum number of objective function evaluations,
            or `None` if no limit is specified.
        :rtype: Optional[int]
        """

    def get_max_time_millis(self) -> Optional[int]:
        """
        Obtain the maximum runtime permitted in milliseconds.

        If no limit is set, `None` is returned.

        :return: the maximum runtime permitted in milliseconds,
            or `None` if no limit is specified.
        :rtype: Optional[int]
        """

    def has_current_best(self) -> bool:  # +book
        """
        Check whethers a current best solution is available.

        As soon as one objective function evaluation has been performed,
        the black-box process can provide a best-so-far solution. Then,
        this method returns True. Otherwise, it returns False.

        :return: True if the current-best solution can be queried.
        :rtype: bool
        """

    def get_current_best_f(self) -> Union[int, float]:  # +book
        """
        Get the objective value of the current best solution.

        :return: the objective value of the current best solution.
        :rtype: Union[int,float]
        """

    def get_copy_of_current_best_x(self, x) -> None:  # +book
        """
        Get a copy of the current best point in the search space.

        :param x: the destination data structure to be overwritten
        """

    def get_copy_of_current_best_y(self, y) -> None:  # +book
        """
        Get a copy of the current best point in the solution space.

        :param y: the destination data structure to be overwritten
        """

    def get_last_improvement_fe(self) -> int:  # +book
        """
        Get the FE at which the last improvement was made.

        :return: the function evaluation when the last improvement was made
        :rtype: int
        :raises ValueError: if no FE was performed yet
        """

    def get_last_improvement_time_millis(self) -> int:
        """
        Get the FE at which the last improvement was made.

        :return: the function evaluation when the last improvement was made
        :rtype: int
        :raises ValueError: if no FE was performed yet
        """

    def get_name(self) -> str:
        """
        Get the name of this process implementation.

        :return: "process"
        """
        return "process"

    def terminate(self) -> None:  # +book
        """
        Terminate this process.

        This function is automatically called at the end of the `with`
        statement, but can also be called by the algorithm when it is
        finished and is also invoked automatically when a termination
        criterion is hit.
        After the first time this method is invoked, :meth:should_terminate`
        becomes `True`.
        """
        self._terminated = True  # +book

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


def check_max_fes(max_fes: Optional[int],
                  none_is_ok: bool = False) -> Optional[int]:
    """
    Check the maximum FEs.

    :param Optional[int] max_fes: the maximum FEs
    :param bool none_is_ok: is None ok?
    :return: the maximum fes, or None
    :rtype: Optional[int]
    :raises TypeError: if max_fes is None or not an int
    :raises ValueError: if max_fes is invalid
    """
    if max_fes is None:
        if none_is_ok:
            return None
        raise TypeError("Maximum FEs must not be None.")

    if not isinstance(max_fes, int):
        raise TypeError("Maximum number of function evaluations must be "
                        f"int, but is {type(max_fes)}.")
    if max_fes <= 0:
        raise ValueError(f"Maximum FEs must be positive, but are {max_fes}.")
    return max_fes


def check_max_time_millis(max_time_millis: Optional[int],
                          none_is_ok: bool = False) -> Optional[int]:
    """
    Check the maximum time in milliseconds.

    :param Optional[int] max_time_millis: the maximum time in milliseconds
    :param bool none_is_ok: is None ok?
    :return: the maximum time in millseconds, or None
    :rtype: Optional[int]
    :raises TypeError: if max_time_millis is None or not an int
    :raises ValueError: if max_time_millis is invalid
    """
    if max_time_millis is None:
        if none_is_ok:
            return None
        raise TypeError("Maximum time in milliseconds must not be None.")

    if not isinstance(max_time_millis, int):
        raise TypeError("Maximum time in milliseconds must be int, but is "
                        f"{type(max_time_millis)}.")
    if max_time_millis <= 0:
        raise ValueError("Maximum time in milliseconds must be positive, "
                         f"but is {max_time_millis}.")
    return max_time_millis


def check_goal_f(goal_f: Union[int, float, None],
                 none_is_ok: bool = False) -> Union[int, float, None]:
    """
    Check the goal objective value.

    :param Optional[int] goal_f: the goal objective value
    :param bool none_is_ok: is None ok?
    :return: the goal objective value, or None
    :rtype: Union[int, float, None]
    :raises TypeError: if goal_f is None or neither an int nor a float
    :raises ValueError: if goal_f is invalid
    """
    if goal_f is None:
        if none_is_ok:
            return None
        raise TypeError("Goal objective value cannot be None.")

    if not (isinstance(goal_f, (int, float))):
        raise TypeError("Goal objective value must be int or float, but is "
                        f"{type(goal_f)}.")
    if isnan(goal_f):
        raise ValueError("Goal objective value must not be NaN, but is "
                         f"{goal_f}.")
    if goal_f >= inf:
        raise ValueError("Goal objective value must be less than positive "
                         f"infinity, but is {goal_f}.")
    return goal_f
