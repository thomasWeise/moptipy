"""Test stuff on real vectors."""

from typing import Callable, Final, Iterable

from numpy.random import Generator, default_rng

from moptipy.api.algorithm import Algorithm
from moptipy.api.objective import Objective
from moptipy.examples.vectors.ackley import Ackley
from moptipy.spaces.bounded_vectorspace import BoundedVectorSpace
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.tests.algorithm import validate_algorithm
from moptipy.utils.types import type_error

#: The dimensions for tests
DIMENSIONS_FOR_TESTS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 10)


def vectors_for_tests() -> Iterable[VectorSpace]:
    """
    Get a sequence of vector spaces for tests.

    :returns: the sequence of vector spaces
    """
    random: Final[Generator] = default_rng()
    spaces: Final[list[VectorSpace]] = []
    for dim in DIMENSIONS_FOR_TESTS:
        if random.integers(2) <= 0:
            spaces.append(VectorSpace(dim))
        else:
            a = b = float(random.uniform(-100.0, 100.0))
            while abs(a - b) < 1e-3:
                b = float(random.uniform(-100.0, 100.0))
            if a > b:
                a, b = b, a
            spaces.append(BoundedVectorSpace(dim, a, b))
    return spaces


def validate_algorithm_on_vectors(
        objective: Objective | Callable[[VectorSpace], Objective],
        algorithm: Algorithm | Callable[[VectorSpace, Objective], Algorithm],
        max_fes: int = 100,
        uses_all_fes_if_goal_not_reached=True) -> None:
    """
    Check the validity of a black-box algorithm on vector problems.

    :param algorithm: the algorithm or algorithm factory
    :param objective: the objective function or function factory
    :param max_fes: the maximum number of FEs
    :param uses_all_fes_if_goal_not_reached: will the algorithm use all FEs
        unless it reaches the goal?
    """
    if not (isinstance(algorithm, Algorithm) or callable(algorithm)):
        raise type_error(algorithm, "algorithm", Algorithm, True)
    if not (isinstance(objective, Objective) or callable(objective)):
        raise type_error(objective, "objective", Objective, True)

    for space in vectors_for_tests():
        if callable(objective):
            objf = objective(space)
            if not isinstance(objf, Objective):
                raise type_error(objf, "result of callable objective",
                                 Objective)
        else:
            objf = objective
        if callable(algorithm):
            algo = algorithm(space, objf)
            if not isinstance(algo, Algorithm):
                raise type_error(algo, "result of callable algorithm",
                                 Algorithm)
        else:
            algo = algorithm

        validate_algorithm(
            algorithm=algo, solution_space=space, objective=objf,
            max_fes=max_fes,
            uses_all_fes_if_goal_not_reached=uses_all_fes_if_goal_not_reached)


def validate_algorithm_on_ackley(
        algorithm: Algorithm | Callable[[VectorSpace, Objective], Algorithm],
        uses_all_fes_if_goal_not_reached: bool = True) -> None:
    """
    Check the validity of a black-box algorithm on Ackley's function.

    :param algorithm: the algorithm or algorithm factory
    :param uses_all_fes_if_goal_not_reached: will the algorithm use all FEs
        unless it reaches the goal?
    """
    validate_algorithm_on_vectors(
        Ackley(), algorithm,
        uses_all_fes_if_goal_not_reached=uses_all_fes_if_goal_not_reached)
