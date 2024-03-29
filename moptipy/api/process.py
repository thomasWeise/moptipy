"""
Processes offer data to both the user and the optimization algorithm.

They provide the information about the optimization process and its current
state as handed to the optimization algorithm and, after the algorithm has
finished, to the user. They also supply the optimization algorithm with
everything it needs to run, e.g., random numbers
(:meth:`~moptipy.api.process.Process.get_random`), they evaluate solutions
(:meth:`~moptipy.api.process.Process.evaluate`) , and they tell it when to
stop (:meth:`~moptipy.api.process.Process.should_terminate`).

The idea behind this interface is to treat optimization algorithms as
so-called *anytime algorithms*. An anytime algorithm will begin with a guess
about what the solution for a problem could be. It will then iteratively
sample and evaluate (:meth:`~moptipy.api.process.Process.evaluate`) new
solutions, i.e., new and hopefully better guesses. It can be stopped at any
time, e.g., by the termination criterion,
:meth:`~moptipy.api.process.Process.should_terminate` and then return the best
guess of the solution (:meth:`~moptipy.api.process.Process.get_copy_of_best_y`,
:meth:`~moptipy.api.process.Process.get_best_f`).

The process API also collects all the information about the optimization
process, performs in-memory logging if wanted, and can write a standardized,
text-based log file for each run of an algorithm in a clear folder structure.
By storing information about the algorithm, the problem, and the system, as
well as the random seed, this allows for self-documenting and replicable
experiments.

The class :class:`Process` is a base class from which all optimization
processes are derived. It is for the standard single-objective optimization
case. A multi-objective variant is given in module
:mod:`~moptipy.api.mo_process` as class
:class:`~moptipy.api.mo_process.MOProcess`.

Furthermore, processes also lent themselves to "forking" off some of the
computational budget of an algorithm to sub-algorithms. For this purpose, the
module :mod:`~moptipy.api.subprocesses` provides specialized routines, such as
:func:`~moptipy.api.subprocesses.for_fes` for creating sub-processes that
forward all method calls to the original process but will perform at most a
given number of objective function evaluations or
:func:`~moptipy.api.subprocesses.from_starting_point`, which creates a
sub-process that has the current-best solution pre-set to a given point in the
search space and its quality.
:func:`~moptipy.api.subprocesses.without_should_terminate` wraps a process in
such a way that the termination criterion
:meth:`~moptipy.api.process.Process.should_terminate`, which is suitable for
invoking externally implemented optimization algorithms that do not know/care
about the `moptipy` API.

1. Mark S. Boddy and Thomas L. Dean. *Solving Time-Dependent Planning
   Problems.* Report CS-89-03, February 1989. Providence, RI, USA: Brown
   University, Department of Computer Science.
   ftp://ftp.cs.brown.edu/pub/techreports/89/cs89-03.pdf
"""
from contextlib import AbstractContextManager
from math import inf, isnan

from numpy.random import Generator
from pycommons.types import check_int_range, type_error

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

        The optimization algorithm and all of its components must only use
        this random number generator for all their non-deterministic
        decisions. In order to guarantee reproducible runs, there must not be
        any other source of randomness. This generator can be seeded in the
        :meth:`~moptipy.api.execution.Execution.set_rand_seed` method of the
        :class:`~moptipy.api.execution.Execution` builder object.

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

        Generally, the termination criterion is configured by the methods
        :meth:`~moptipy.api.execution.Execution.set_max_fes`,
        :meth:`~moptipy.api.execution.Execution.set_max_time_millis`, and
        :meth:`~moptipy.api.execution.Execution.set_goal_f` of the
        :class:`~moptipy.api.execution.Execution` builder. Furthermore, if
        the objective function has a finite
        :meth:`~moptipy.api.objective.Objective.lower_bound`, then this lower
        bound is also used as goal objective value if no goal objective value
        is specified via :meth:`~moptipy.api.execution.Execution.set_goal_f`.
        :meth:`should_terminate` then returns `True` as soon as any one of the
        configured criteria is met, i.e., the process terminates when the
        earliest one of the criteria is met.

        :return: `True` if the process should terminate, `False` if not
        """

    def evaluate(self, x) -> float | int:  # +book
        """
        Evaluate a solution `x` and return its objective value.

        This method implements the
        :meth:`~moptipy.api.objective.Objective.evaluate` method of
        the :class:`moptipy.api.objective.Objective` function interface,
        but on :class:`Process` level.

        The return value is either an integer or a float and must be
        finite. Smaller objective values are better, i.e., all objective
        functions are subject to minimization.

        This method here is usually a wrapper that internally invokes the
        actual :class:`~moptipy.api.objective.Objective` function, but it does
        more: While it does use the
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
        moves, in memory if logging is activated. (See
        :meth:`~moptipy.api.execution.Execution.set_log_file`,
        :meth:`~moptipy.api.execution.Execution.set_log_improvements`, and
        :meth:`~moptipy.api.execution.Execution.set_log_all_fes`.)

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

    def register(self, x, f: int | float) -> None:
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

    def get_max_fes(self) -> int | None:
        """
        Obtain the maximum number of permitted objective function evaluations.

        If no limit is set, `None` is returned.

        :return: the maximum number of objective function evaluations,
            or `None` if no limit is specified.
        """

    def get_max_time_millis(self) -> int | None:
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

        See Also
            - :meth:`get_best_f`
            - :meth:`get_copy_of_best_x`
            - :meth:`get_copy_of_best_y`
        """

    def get_best_f(self) -> int | float:  # +book
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

        See Also
            - :meth:`has_best`
            - :meth:`get_copy_of_best_x`
            - :meth:`get_copy_of_best_y`
        """

    def get_copy_of_best_x(self, x) -> None:  # +book
        """
        Get a copy of the current best point in the search space.

        This always corresponds to the point in the search space encoding the
        best-so-far solution, i.e., the best point in the search space that
        you have passed to :meth:`evaluate` or :meth:`register` so far.
        It is *NOT* the best global optimum for the optimization problem. It
        corresponds to the best solution that the process has seen *so far*,
        the current best solution.

        Even if the optimization algorithm using this process does not
        preserve this solution in special variable and has already lost it
        again, this method will still return it. The optimization process
        encapsulated by this `process` object will always remember it.

        This also means that your algorithm implementations do not need to
        store the best-so-far solution anywhere if doing so would be
        complicated. They can obtain it simply from this method whenever
        needed.

        You should only call this method if you are either sure that you
        have invoked :meth:`evaluate` before :meth:`register` of if you called
        :meth:`has_best` before and it returned `True`.

        For understanding the relationship between the search space and the
        solution space, see module :mod:`~moptipy.api.encoding`.

        :param x: the destination data structure to be overwritten

        See Also
            - :meth:`has_best`
            - :meth:`get_best_f`
            - :meth:`get_copy_of_best_y`
        """

    def get_copy_of_best_y(self, y) -> None:  # +book
        """
        Get a copy of the current best point in the solution space.

        This always corresponds to the best-so-far solution, i.e., the
        best solution that you have passed to :meth:`evaluate` or
        :meth:`register` so far. It is *NOT* the global optimum for the
        optimization problem. It is the best solution that the process has
        seen *so far*, the current best solution.

        You should only call this method if you are either sure that you
        have invoked meth:`evaluate` before :meth:`register` of if you called
        :meth:`has_best` before and it returned `True`.

        :param y: the destination data structure to be overwritten

        See Also
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

        This method is overwritten for each subclass of :class:`Process`
        and then returns a short descriptive value of these classes.

        :return: "process" for this base class
        """
        return "process"

    def terminate(self) -> None:  # +book
        """
        Terminate this process.

        This function is automatically called at the end of the `with`
        statement, but can also be called by the algorithm when it is
        finished and is also invoked automatically when a termination
        criterion is hit.
        After the first time this method is invoked, :meth:`should_terminate`
        becomes `True`.
        """

    def has_log(self) -> bool:
        """
        Will any information of this process be logged?.

        Only if this method returns `True`, invoking :meth:`add_log_section`
        makes any sense. Otherwise, the data would just be discarded.

        :retval `True`: if the process is associated with a log output
        :retval `False`: if no information is stored in a log output
        """

    def add_log_section(self, title: str, text: str) -> None:
        """
        Add a section to the log, if a log is written (otherwise ignore it).

        When creating the experiment
        :class:`~moptipy.api.execution.Execution`, you can specify a log file
        via method :meth:`~moptipy.api.execution.Execution.set_log_file`.
        Then, the results of your algorithm and the system configuration will
        be stored as text in this file. Each type of information will be
        stored in a different section. The end state with the final solution
        quality, for instance, will be stored in a section named `STATE`.
        Each section begins with the line `BEGIN_XXX` and ends with the line
        `END_XXX`, where `XXX` is the name of the section. Between these two
        lines, all the contents of the section are stored.

        This method here allows you to add a custom section to your log file.
        This can happen in your implementation of the method
        :meth:`~moptipy.api.algorithm.Algorithm.solve` of your algorithm.
        (Ideally at its end.) Of course, invoking this method only makes sense
        if there actually is a log file. You can check for this by calling
        :meth:`has_log`.

        You can specify a custom section name (which must be in upper case
        characters) and a custom section body text.
        Of course, the name of this section must not clash with any other
        section name. Neither the section name nor section body should contain
        strings like `BEGIN_` or `END_`, and such and such. You do not want to
        mess up your log files. Ofcourse you can add a section with a given
        name only once, because otherwise there would be a name clash.
        Anyway, if you add sections like this, they will be appended at the
        end of the log file. This way, you have all the standard log data and
        your additional information in one consistent file.

        Be advised: Adding sections costs time and memory. You do not want to
        do such a thing in a loop. If your algorithm should store additional
        data, it makes sense to gather this data in an efficient way during
        the run and only flush it to a section at the end of the run.

        :param title: the title of the log section
        :param text: the text to log
        """

    def get_log_basename(self) -> str | None:
        """
        Get the basename of the log, if any.

        If a log file is associated with this process, then this function
        returns the name of the log file without the file suffix. If no log
        file is associated with the process, then `None` is returned.

        This can be used to store additional information during the run of
        the optimization algorithm. However, treat this carefully, as some
        files with the same base name may exist or be generated by other
        modules.

        :returns: the path to the log file without the file suffix if a log
            file is associated with the process, or `None` otherwise
        """
        return None

    def initialize(self) -> None:
        """
        Raise an error because this method shall never be called.

        :raises ValueError: always
        """
        raise ValueError("Never call the initialize() method of a Process!")

    def __enter__(self) -> "Process":
        """
        Begin a `with` statement.

        :return: this process itself
        """
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> bool:
        """
        End a `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        :returns: `True` to suppress an exception, `False` to rethrow it
        """
        self.terminate()
        return exception_type is None


def check_max_fes(max_fes: int | None,
                  none_is_ok: bool = False) -> int | None:
    """
    Check the maximum FEs.

    This is a small utility method that validates whether a maximum for the
    objective function evaluations (FEs) is valid.

    :param max_fes: the maximum FEs
    :param none_is_ok: is `None` ok?
    :return: the maximum fes, or `None`
    :raises TypeError: if `max_fes` is `None` (and `None` is not allowed) or
        not an `int`
    :raises ValueError: if `max_fes` is invalid
    """
    if max_fes is None:
        if none_is_ok:
            return None
        raise type_error(max_fes, "max_fes", int)
    return check_int_range(max_fes, "max_fes", 1, 1_000_000_000_000_000)


def check_max_time_millis(max_time_millis: int | None,
                          none_is_ok: bool = False) -> int | None:
    """
    Check the maximum time in milliseconds.

    This is a small utility method that validates whether a maximum for the
    milliseconds that can be used as runtime limit is valid.

    :param max_time_millis: the maximum time in milliseconds
    :param none_is_ok: is None ok?
    :return: the maximum time in milliseconds, or `None`
    :raises TypeError: if `max_time_millis` is `None` (and `None` is not
        allowed) or not an `int`
    :raises ValueError: if `max_time_millis` is invalid
    """
    if max_time_millis is None:
        if none_is_ok:
            return None
        raise type_error(max_time_millis, "max_time_millis", int)
    return check_int_range(
        max_time_millis, "max_time_millis", 1, 100_000_000_000)


def check_goal_f(goal_f: int | float | None,
                 none_is_ok: bool = False) -> int | float | None:
    """
    Check the goal objective value.

    This is a small utility method that validates whether a goal objective
    value is valid.

    :param goal_f: the goal objective value
    :param none_is_ok: is `None` ok?
    :return: the goal objective value, or `None`
    :raises TypeError: if `goal_f` is `None` (and `None` is not allowed) or
        neither an `int` nor a `float`
    :raises ValueError: if `goal_f` is invalid
    """
    if not (isinstance(goal_f, int | float)):
        if none_is_ok and (goal_f is None):
            return None
        raise type_error(goal_f, "goal_f", (int, float))
    if isnan(goal_f):
        raise ValueError("Goal objective value must not be NaN, but is "
                         f"{goal_f}.")
    if goal_f >= inf:
        raise ValueError("Goal objective value must be less than positive "
                         f"infinity, but is {goal_f}.")
    return goal_f
