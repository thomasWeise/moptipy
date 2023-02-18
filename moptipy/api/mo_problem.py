"""
The base classes for multi-objective optimization problems.

This class provides the ability to evaluate solutions according to multiple
criteria. The evaluation results are stored in a numpy array and also are
scalarized to a single value.

Basically, a multi-objective problem provides three essential components:

1. It can evaluate a candidate solution according to multiple optimization
   objectives. Each objective returns one value, subject to minimization,
   and all the values are stored in a single numpy array.
   This is done by :meth:`~moptipy.api.mo_problem.MOProblem.f_evaluate`
2. It provides a criterion deciding whether one such objective vector
   dominates (i.e., is strictly better than) another one. This is done by
   :meth:`~moptipy.api.mo_problem.MOProblem.f_dominates`. The default
   definition adheres to the standard "domination" definition in
   multi-objective optimization: A vector `a` dominates a vector `b` if it
   is not worse in any objective value and better in at least one. But if
   need be, you can overwrite this behavior.
3. A scalarization approach: When evaluating a solution, the result is not
   just the objective vector itself, but also a single scalar value. This is
   needed to create compatibility to single-objective optimization. Matter of
   fact, a :class:`~moptipy.api.mo_problem.MOProblem` is actually a subclass
   of :class:`~moptipy.api.objective.Objective`. This means that via this
   scalarization, all multi-objective problems can also be considered as
   single-objective problems. This means that single-objective algorithms can
   be applied to them as-is. It also means that log files are compatible.
   Multi-objective algorithms can just ignore the scalarization result and
   focus on the domination relationship. Often, a weighted sum approach
   (:class:`~moptipy.mo.problem.weighted_sum.WeightedSum`) may be the method
   of choice for scalarization.
"""
from typing import Any, Final

import numpy as np

from moptipy.api.logging import KEY_SPACE_NUM_VARS, SCOPE_OBJECTIVE_FUNCTION
from moptipy.api.mo_utils import dominates
from moptipy.api.objective import Objective, check_objective
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import (
    DEFAULT_FLOAT,
    DEFAULT_INT,
    DEFAULT_UNSIGNED_INT,
    KEY_NUMPY_TYPE,
    int_range_to_dtype,
    numpy_type_to_str,
)
from moptipy.utils.types import type_error


class MOProblem(Objective):
    """
    The base class for multi-objective optimization problems.

    A multi-objective optimization problem is defined as a set of
    :class:`~moptipy.api.objective.Objective` functions. Each candidate
    solution is evaluated using each of the objectives, i.e., is rated by a
    vector of objective values. This vector is the basis for deciding which
    candidate solutions to keep and which to discard.

    In multi-objective optimization, this decision is based on "domination."
    A solution `a` dominates a solution `b` if it is not worse in any
    objective and better in at least one. This comparison behavior is
    implemented in method
    :meth:`~moptipy.api.mo_problem.MOProblem.f_dominates` and can be
    overwritten if need be.

    In our implementation, we prescribe that each multi-objective optimization
    problem must also be accompanied by a scalarization function, i.e., a
    function that represents the vector of objective values as a single scalar
    value. The whole multi-objective problem can then be viewed also as a
    single objective function itself. The method
    :meth:`~moptipy.api.mo_problem.MOProblem.evaluate` first evaluates all of
    the objective functions and obtains the vector of objective values. It then
    scalarizes the result into a single scalar quality and returns it.
    Multi-objective algorithms may instead use the method
    :meth:`~moptipy.api.mo_problem.MOProblem.f_evaluate`, which also allows
    a vector to be passed in which will then be filled with the results of the
    individual objective functions.

    This makes multi-objective optimization with moptipy compatible with
    single-objective optimization. In other words, all optimization methods
    implemented for single-objective processes
    :class:`~moptipy.api.process.Process` will work out-of-the-box with the
    multi-objective version :class:`~moptipy.api.mo_process.MOProcess`.

    Warning: We use instances of :class:`numpy.ndarray` to represent the
    vectors of objective values. This necessitates that each objective
    function has, if it is integer-valued
    (:meth:`~moptipy.api.objective.Objective.is_always_integer` is `True`)
    a range that fits well into at least a 64-bit integer. Specifically, it
    must be possible to compute "a - b" without overflow or loss of sign for
    any two objective values "a" and "b" within the confines of a numpy
    signed 64-bit integer.
    """

    def f_create(self) -> np.ndarray:
        """
        Create a vector to receive the objective values.

        This array will be of the length returned by :meth:`f_dimension` and
        of the `dtype` of :meth:`f_dtype`.

        :returns: a vector to receive the objective values
        """
        return np.empty(self.f_dimension(), self.f_dtype())

    def f_dtype(self) -> np.dtype:
        """
        Get the data type used in :meth:`f_create`.

        This data type will be an integer data type if all the objective
        functions are integer-valued. If the bounds of the objective values
        are known, then this type will be "big enough" to allow the
        subtraction "a - b" of any two objective vectors "a" and "b" to be
        computed without overflow or loss of sign. At most, however, this
        data type will be a 64-bit integer.
        If any one of the objective functions returns floating point data,
        this data type will be a floating point type.

        :returns: the data type used by :meth:`f_create`.
        """

    def f_dimension(self) -> int:
        """
        Obtain the number of objective functions.

        :returns: the number of objective functions
        """

    def f_validate(self, x: np.ndarray) -> None:
        """
        Validate the objective vector.

        :param x: the numpy vector
        :raises TypeError: if the string is not an element of this space.
        :raises ValueError: if the shape of the vector is wrong or any of its
            element is not finite.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(x, "x", np.ndarray)
        shape = x.shape
        if len(shape) != 1:
            raise ValueError(
                f"{x} cannot have more than one dimension, but has {shape}!")
        dim = self.f_dimension()  # pylint: disable=E1111
        if shape[0] != dim:
            raise ValueError(
                f"{x} should have length {dim} but has {shape[0]}!")
        dt = self.f_dtype()  # pylint: disable=E1111
        if x.dtype != dt:
            raise ValueError(f"{x} should have dtype {dt} but has {x.dtype}!")

    def f_evaluate(self, x, fs: np.ndarray) -> int | float:
        """
        Perform the multi-objective evaluation of a solution.

        This method fills the objective vector `fs` with the results of the
        objective functions evaluated on `x`. It then returns the scalarized
        result, i.e., a single scalar value computed based on all values
        in `fs`.

        :param x: the solution to be evaluated
        :param fs: the array to receive the objective values
        :returns: the scalarization result
        """

    # noinspection PyMethodMayBeStatic
    def f_dominates(self, a: np.ndarray, b: np.ndarray) -> int:
        """
        Check if an objective vector dominates or is dominated by another one.

        Usually, one vector is said to dominate another one if it is not worse
        in any objective and better in at least one. This behavior is
        implemented in :func:`moptipy.api.mo_utils.dominates` and this is also
        the default behavior of this method. However, depending on your
        concrete optimization task, you may overwrite this behavior.

        :param a: the first objective vector
        :param b: the second objective value
        :returns: an integer value indicating the domination relationship
        :retval -1: if `a` dominates `b`
        :retval 1: if `b` dominates `a`
        :retval 2: if `b` equals `a`
        :retval 0: if `a` and `b` are mutually non-dominated, i.e., if neither
            `a` dominates `b` not `b` dominates `a` and `b` is also different
            from `a`
        """
        return dominates(a, b)

    def evaluate(self, x) -> float | int:
        """
        Evaluate a solution `x` and return its scalarized objective value.

        This method computes all objective values for a given solution and
        then returns the scalarized result. The objective values themselves
        are directly discarted and not used. It makes a multi-objective
        problem compatible with single-objective optimization.

        :param x: the candidate solution
        :returns: the scalarized objective value
        """
        return self.f_evaluate(x, self.f_create())

    def __str__(self) -> str:
        """
        Get the string representation of this multi-objective problem.

        :returns: the string representation of this multi-objective problem
        """
        return "moProblem"


def check_mo_problem(mo_problem: Any) -> MOProblem:
    """
    Check whether an object is a valid instance of :class:`MOProblem`.

    :param mo_problem: the multi-objective optimization problem
    :return: the mo-problem
    :raises TypeError: if `mo_problem` is not an instance of
        :class:`MOProblem`

    >>> check_mo_problem(MOProblem())
    moProblem
    >>> try:
    ...     check_mo_problem(1)
    ... except TypeError as te:
    ...     print(te)
    multi-objective optimziation problem should be an instance of moptipy.\
api.mo_problem.MOProblem but is int, namely '1'.
    >>> try:
    ...     check_mo_problem(None)
    ... except TypeError as te:
    ...     print(te)
    multi-objective optimziation problem should be an instance of moptipy.\
api.mo_problem.MOProblem but is None.
    """
    if isinstance(mo_problem, MOProblem):
        return mo_problem
    raise type_error(mo_problem,
                     "multi-objective optimziation problem", MOProblem)


class MOSOProblemBridge(MOProblem):
    """A bridge between multi-objective and single-objective optimization."""

    def __init__(self, objective: Objective) -> None:
        """Initialize the bridge."""
        super().__init__()
        check_objective(objective)

        self.evaluate = objective.evaluate  # type: ignore
        self.lower_bound = objective.lower_bound  # type: ignore
        self.upper_bound = objective.upper_bound  # type: ignore
        self.is_always_integer = objective.is_always_integer  # type: ignore

        dt: np.dtype
        if self.is_always_integer():
            lb: int | float = self.lower_bound()
            ub: int | float = self.upper_bound()
            dt = DEFAULT_INT
            if isinstance(lb, int):
                if isinstance(ub, int):
                    dt = int_range_to_dtype(lb, ub)
                elif lb >= 0:
                    dt = DEFAULT_UNSIGNED_INT
        else:
            dt = DEFAULT_FLOAT

        #: the data type of the objective array
        self.__dtype: Final[np.dtype] = dt
        #: the objective function
        self.__f: Final[Objective] = objective
        self.f_create = lambda dd=dt: np.empty(1, dd)  # type: ignore
        self.f_dimension = lambda: 1  # type: ignore

    def initialize(self) -> None:
        """Initialize the MO-problem bridge."""
        super().initialize()
        self.__f.initialize()

    def f_evaluate(self, x, fs: np.ndarray) -> int | float:
        """
        Evaluate the candidate solution.

        :param x: the solution
        :param fs: the objective vector, will become `[res]`
        :returns: the objective value `res`
        """
        res: Final[int | float] = self.evaluate(x)
        fs[0] = res
        return res

    def f_dtype(self) -> np.dtype:
        """Get the objective vector dtype."""
        return self.__dtype

    def f_validate(self, x: np.ndarray) -> None:
        """
        Validate the objective vector.

        :param x: the numpy array with the objective values
        """
        if not isinstance(x, np.ndarray):
            raise type_error(x, "x", np.ndarray)
        if len(x) != 1:
            raise ValueError(f"length of x={len(x)}")
        lb = self.lower_bound()
        ub = self.upper_bound()
        if not (lb <= x[0] <= ub):
            raise ValueError(f"failed: {lb} <= {x[0]} <= {ub}")

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this function to the provided destination.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_SPACE_NUM_VARS, "1")
        logger.key_value(KEY_NUMPY_TYPE, numpy_type_to_str(self.__dtype))
        with logger.scope(f"{SCOPE_OBJECTIVE_FUNCTION}{0}") as scope:
            self.__f.log_parameters_to(scope)
