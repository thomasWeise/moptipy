"""
The base classes for multi-objective optimization problems.

This class provides the ability to evaluate solutions according to multiple
criteria. The evaluation results are stored in a numpy array and also are
scalarized to a single value.

An :class:`~moptipy.api.mo_problem.MOProblem` furthermore also exhibits
:class:`~moptipy.api.space.Space`-like behavior for instantiating and
processing such objective vectors.
"""
from typing import Union

import numpy as np

from moptipy.api.objective import Objective, check_objective
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
