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
from moptipy.utils.strings import float_to_str
from moptipy.utils.types import type_error


class MOProblem(Objective):
    """
    The base class for multi-objective optimization problems.

    A multi-objective optimization problem is defined as a set of
    :class:`~moptipy.api.objective.Objective` functions. Each candidate
    solution is evaluated using each of the objectives, i.e., is rated by a
    vector of objective values. This vector is the basis for deciding which
    candidate solutions to keep and which to discard.

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

        :returns: a vector to receive the objective values
        """

    def f_dimension(self) -> int:
        """
        Obtain the number of objective functions.

        :returns: the number of objective functions
        """

    def f_to_str(self, x: np.ndarray) -> str:
        """
        Convert an objective vector to a string, using `,` as separator.

        :param x: the objective vector
        :return: the string
        """

    def f_validate(self, x: np.ndarray) -> None:
        """
        Validate the objective vector.

        :param x: the numpy vector
        :raises TypeError: if the string is not an element of this space.
        :raises ValueError: if the shape of the vector is wrong or any of its
            element is not finite.
        """

    def f_copy(self, dest, source) -> None:
        """
        Copy one objective vector to another one.

        :param dest: the destination objective vector,
            whose contents will be overwritten with those from `source`
        :param source: the source objective vector, which remains
            unchanged and whose contents will be copied to `dest`
        """

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


def _cpy(dest: np.ndarray, src: np.ndarray) -> None:
    """
    Copy a numpy array of length 1.

    :param dest: the destination
    :param src: the source
    """
    dest[0] = src[0]


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
            self.f_to_str = lambda x: str(int(x[0]))  # type: ignore
        else:
            dt = DEFAULT_FLOAT
            self.f_to_str = lambda x: float_to_str(  # type: ignore
                float(x[0]))  # type: ignore

        #: the data type of the objective array
        self.__dtype: Final[np.dtype] = dt
        #: the objective function
        self.__f: Final[Objective] = objective
        self.f_create = lambda dd=dt: np.empty(1, dd)  # type: ignore
        self.f_dimension = lambda: 1  # type: ignore
        self.f_copy = _cpy  # type: ignore

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
