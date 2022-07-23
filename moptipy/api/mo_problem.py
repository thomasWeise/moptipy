"""
The base classes for multi-objective optimization problems.

This class provides the ability to evaluate solutions according to multiple
criteria. The evaluation results are stored in a numpy array and also are
scalarized to a single value.

An :class:`~moptipy.api.mo_problem.MOProblem` furthermore also exhibits
:class:`~moptipy.api.space.Space`-like behavior for instantiating and
processing such objective vectors.
"""
from typing import Union, Final

import numpy as np

from moptipy.api.logging import KEY_SPACE_NUM_VARS
from moptipy.api.logging import SCOPE_OBJECTIVE_FUNCTION
from moptipy.api.objective import Objective, check_objective
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_INT, DEFAULT_UNSIGNED_INT, \
    DEFAULT_FLOAT, int_range_to_dtype, KEY_NUMPY_TYPE, val_numpy_type
from moptipy.utils.types import type_error


class MOProblem(Objective):
    """
    The base class for multi-objective optimization problems.

    A multi-objective optimization problem is defined as a set of
    :class:`~moptipy.api.objective.Objective` functions. Each candidate
    solution is evaluated using each of the objectives, i.e., is rated by a
    vector of objective values. This vector is the basis for deciding which
    candidate solutions to keep and which to discard.

    Warning: We use instances of :class:`numpy.ndarray` to represent the
    vectors of objective values. This necessitates that each objective
    function has, if it is integer-valued
    (:meth:`~moptipy.api.objective.Objective.is_always_integer` is `True`)
    a range that fits well into at least a 64-bit integer. Specifically, it
    must be possible to compute "a - b" without overflow or loss of sign for
    any two objective values "a" and "b" within the confines of a numpy
    signed 64-bit integer.

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

    def f_evaluate(self, x, fs: np.ndarray) -> Union[int, float]:
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

    def evaluate(self, x) -> Union[float, int]:
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
        """Get the string representation of this scalarization."""
        return "moProblem"


def check_mo_problem(mo_problem: MOProblem) -> MOProblem:
    """
    Check whether an object is a valid instance of :class:`MOProblem`.

    :param mo_problem: the multi-objective optimization problem
    :return: the mo-problem
    :raises TypeError: if `mo_problem` is not an instance of
        :class:`MOProblem`
    """
    check_objective(mo_problem)
    if not isinstance(mo_problem, MOProblem):
        raise type_error(mo_problem,
                         "multi-objective optimziation problem", MOProblem)
    return mo_problem


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
            lb: Union[int, float] = self.lower_bound()
            ub: Union[int, float] = self.upper_bound()
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

    def f_evaluate(self, x, fs: np.ndarray) -> Union[int, float]:
        """
        Evaluate the candidate solution.

        :param x: the solution
        :param fs: the objective vector, will become `[res]`
        :returns: the objective value `res`
        """
        res: Final[Union[int, float]] = self.evaluate(x)
        fs[0] = res
        return res

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
        logger.key_value(KEY_NUMPY_TYPE, val_numpy_type(self.__dtype))
        with logger.scope(f"{SCOPE_OBJECTIVE_FUNCTION}{0}") as scope:
            self.__f.log_parameters_to(scope)
