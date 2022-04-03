"""
Processes offer data to both the user and the optimization algorithm.

They provide the information about the optimization process and its current
state as handed to the optimization algorithm and, after the algorithm has
finished, to the user.
"""
from math import inf, isnan
from typing import Optional, Union
from contextlib import AbstractContextManager

from numpy.random import Generator

from moptipy.api.objective import Objective
from moptipy.api.space import Space


# start book
class Process(Space, Objective, AbstractContextManager):
    """
    Processes offer data to the optimization algorithm and the user.

    A Process presents the objective function and search space to an
    optimization algorithm. Since it wraps the actual objective
    function, it can see all evaluated solutions and remember the
    best-so-far solution. It can also count the FEs and the runtime
    that has passed. Therefore, it also presents the termination
    criterion to the optimization algorithm. It also provides a random
    number generator the algorithm. It can write log files with the
    progress of the search and the end result. Finally, it provides
    the end result to the user, who can access it after the algorithm
    has finished.
    """

# end book

    def get_random(self) -> Generator:  # +book
        """
        Obtain the random number generator.

        :return: the random number generator
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
        """

    def evaluate(self, x) -> Union[float, int]:  # +book
        """
        Evaluate a solution `x` and return its objective value.

        This method implements the
        :meth:`~moptipy.api.objective.Objective.evaluate` method of
        the :class:`moptipy.api.objective.Objective` function interface,
        but on :class:`Process` level.

        The return value is either an integer or a float and must be
        finite. Smaller objective values are better, i.e., all objective
        functions are subject to minimization.

        This method here is a wrapper that internally invokes the actual
        :class:`~moptipy.api.objective.Objective` function, but it does more.

        While it does use the
        :meth:`~moptipy.api.objective.Objective.evaluate` method of the
        objective function to compute the quality of a candidate solution,
        it also internally increments the counter for the objective function
        evaluations (FEs) that have passed. You can request the number of
        these FEs via :meth:`get_consumed_fes` (and also the time that has
        passed via :meth:`get_consumed_time_millis`, but this is unrelated
        to the :meth:`evaluate` method).

        Still, counting the FEs like this allows us to know when, e.g., the
        computational budget in terms of a maximum permitted number of FEs
        has been exhausted, in which case :meth:`should_terminate` will
        become `True`.

        Also, since this method will see all objective values and the
        corresponding candidate solutions, it is able to internally remember
        the best solution you have ever created and its corresponding
        objective value. Therefore, the optimization :class:`Process` can
        provide both to you via the methods :meth:`has_best`,
        :meth:`get_copy_of_best_x`, :meth:`get_copy_of_best_y`, and
        :meth:`get_best_f`. At the same time, if a goal objective value or
        lower bound for the objective function is specified and one solution
        is seen that has such a quality, :meth:`should_terminate` will again
        become `True`.

        Finally, this method also performs all logging, e.g., of improving
        moves, in memory if logging is activated.

        In some cases, you may not need to invoke the original objective
        function via this wrapper to obtain the objective value of a solution.
        Indeed, in some cases you *know* the objective value because of the
        way you constructed the solution. However, you still need to tell our
        system the objective value and provide the solution to ensure the
        correct counting of FEs, the correct preservation of the best
        solution, and the correct setting of the termination criterion. For
        these situations, you will call :meth:`register` instead of
        :meth:`evaluate`.

        :param x: the candidate solution
        :return: the objective value
        """

    def register(self, x, f: Union[int, float]) -> None:
        """
        Register a solution `x` with externally-evaluated objective value.

        This function is equivalent to :meth:`evaluate`, but receives the
        objective value as parameter. In some problems, algorithms can compute
        the objective value of a solution very efficiently without passing it
        to the objective function.

        For example, on the Traveling Salesperson Problem with n cities, if
        you have a tour of known length and swap two cities in it, then you
        can compute the overall tour length in O(1) instead of O(n) that you
        would need to pay for a full evaluation. In such a case, you could
        use `register` instead of `evaluate`.

        `x` must be provided if `f` marks an improvement. In this case, the
        contents of `x` will be copied to an internal variable remembering the
        best-so-far solution. If `f` is not an improvement, you may pass in
        `None` for `x` or just any valid point in the search space.

        For each candidate solution you construct, you must call either
        :meth:`evaluate` or :meth:`register`. This is because these two
        functions also count the objective function evaluations (FEs) that
        have passed. This is needed to check the termination criterion, for
        instance.

        :param x: the candidate solution
        :param f: the objective value
        """

    def get_consumed_fes(self) -> int:
        """
        Obtain the number consumed objective function evaluations.

        This is the number of calls to :meth:`evaluate`.

        :return: the number of objective function evaluations so far
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
        """

    def get_max_time_millis(self) -> Optional[int]:
        """
        Obtain the maximum runtime permitted in milliseconds.

        If no limit is set, `None` is returned.

        :return: the maximum runtime permitted in milliseconds,
            or `None` if no limit is specified.
        """

    def has_best(self) -> bool:  # +book
        """
        Check whether a current best solution is available.

        As soon as one objective function evaluation has been performed,
        the black-box process can provide a best-so-far solution. Then,
        this method returns `True`. Otherwise, it returns `False`. This
        means that this method returns `True` if and only if you have
        called either :meth:`evaluate` or :meth:`register` at least once.

        :return: True if the current-best solution can be queried.

        See also:
            - :meth:`get_best_f`
            - :meth:`get_copy_of_best_x`
            - :meth:`get_copy_of_best_y`
        """

    def get_best_f(self) -> Union[int, float]:  # +book
        """
        Get the objective value of the current best solution.

        This always corresponds to the best-so-far solution, i.e., the
        best solution that you have passed to :meth:`evaluate` or
        :meth:`register` so far. It is *NOT* the best possible objective
        value for the optimization problem. It is the best objective value
        that the process has seen *so far*, the current best objective value.

        You should only call this method if you are either sure that you
        have invoked meth:`evaluate` before :meth:`register` of if you called
        :meth:`has_best` before and it returned `True`.

        :return: the objective value of the current best solution.

        See also:
            - :meth:`has_best`
            - :meth:`get_copy_of_best_x`
            - :meth:`get_copy_of_best_y`
        """

    def get_copy_of_best_x(self, x) -> None:  # +book
        """
        Get a copy of the current best point in the search space.

        This always corresponds to the best-so-far solution, i.e., the
        best solution that you have passed to :meth:`evaluate` or
        :meth:`register` so far. It is *NOT* the best possible objective
        value for the optimization problem. It is the best solution that
        the process has seen *so far*, the current best solution.

        You should only call this method if you are either sure that you
        have invoked meth:`evaluate` before :meth:`register` of if you called
        :meth:`has_best` before and it returned `True`.

        :param x: the destination data structure to be overwritten

        See also:
            - :meth:`has_best`
            - :meth:`get_best_f`
            - :meth:`get_copy_of_best_y`
        """

    def get_copy_of_best_y(self, y) -> None:  # +book
        """
        Get a copy of the current best point in the solution space.

        This always corresponds to the best-so-far solution, i.e., the
        best solution that you have passed to :meth:`evaluate` or
        :meth:`register` so far. It is *NOT* the best possible objective
        value for the optimization problem. It is the best solution that
        the process has seen *so far*, the current best solution.

        You should only call this method if you are either sure that you
        have invoked meth:`evaluate` before :meth:`register` of if you called
        :meth:`has_best` before and it returned `True`.

        :param y: the destination data structure to be overwritten

        See also:
            - :meth:`has_best`
            - :meth:`get_best_f`
            - :meth:`get_copy_of_best_x`
        """

    def get_last_improvement_fe(self) -> int:  # +book
        """
        Get the FE at which the last improvement was made.

        You should only call this method if you are either sure that you
        have invoked meth:`evaluate` before :meth:`register` of if you called
        :meth:`has_best` before and it returned `True`.

        :return: the function evaluation when the last improvement was made
        :raises ValueError: if no FE was performed yet
        """

    def get_last_improvement_time_millis(self) -> int:
        """
        Get the FE at which the last improvement was made.

        You should only call this method if you are either sure that you
        have invoked meth:`evaluate` before :meth:`register` of if you called
        :meth:`has_best` before and it returned `True`.

        :return: the function evaluation when the last improvement was made
        :raises ValueError: if no FE was performed yet
        """

    def __str__(self) -> str:
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

    def __enter__(self) -> 'Process':
        """
        Begin a `with` statement.

        :return: this process itself
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

    :param max_fes: the maximum FEs
    :param none_is_ok: is None ok?
    :return: the maximum fes, or None
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

    :param max_time_millis: the maximum time in milliseconds
    :param none_is_ok: is None ok?
    :return: the maximum time in millseconds, or None
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

    :param goal_f: the goal objective value
    :param none_is_ok: is None ok?
    :return: the goal objective value, or None
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
