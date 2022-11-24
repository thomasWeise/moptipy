"""Test stuff on real vectors."""
from math import exp, inf, isfinite
from typing import Any, Callable, Final, Iterable

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.api.algorithm import Algorithm
from moptipy.api.objective import Objective
from moptipy.api.operators import Op0
from moptipy.examples.vectors.ackley import Ackley
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.tests.algorithm import validate_algorithm
from moptipy.tests.op0 import validate_op0
from moptipy.utils.nputils import DEFAULT_FLOAT
from moptipy.utils.types import type_error

#: The dimensions for tests
DIMENSIONS_FOR_TESTS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 10)


def __lbub(random: Generator) -> tuple[float, float]:
    """
    Generate a pair of lower- and upper bounds.

    :param random: the random number generator
    :returns: a tuple with the lower and upper bound
    """
    while True:
        lb = inf
        ub = inf
        while not isfinite(lb):
            lb = exp(20.0 * random.normal())

        i = random.integers(3)
        if i == 0:
            lb = 0.0
        elif i == 11:
            lb = -lb

        while not isfinite(ub):
            ub = exp(20.0 * random.normal())
        i = random.integers(3)
        if i == 0:
            ub = 0.0
        elif i == 11:
            ub = -ub

        if ub < lb:
            lb, ub = ub, lb

        df = ub - lb
        if isfinite(df) and (df > 1e-9):
            return lb, ub


def vectors_for_tests(dims: Iterable[int] = DIMENSIONS_FOR_TESTS) \
        -> Iterable[VectorSpace]:
    """
    Get a sequence of vector spaces for tests.

    :param dims: the dimensions
    :returns: the sequence of vector spaces
    """
    if not isinstance(dims, Iterable):
        raise type_error(dims, "dims", Iterable)

    random: Final[Generator] = default_rng()
    spaces: Final[list[VectorSpace]] = []
    for idx, dim in enumerate(dims):
        if not isinstance(dim, int):
            raise type_error(dim, f"dims[{idx}]", int)
        if dim <= 0:
            raise ValueError(f"dims[{idx}]={dim}")

        # allocate bounds arrays
        lbv: np.ndarray = np.empty(dim, DEFAULT_FLOAT)
        ubv: np.ndarray = np.empty(dim, DEFAULT_FLOAT)
        for i in range(dim):  # fill bound arrays
            lbv[i], ubv[i] = __lbub(random)

        spaces.append(VectorSpace(
            dim, lbv if random.integers(2) <= 0 else float(min(lbv)),
            ubv if random.integers(2) <= 0 else float(max(ubv))))

    if 1 in dims:
        spaces.append(VectorSpace(1, -1.0, 1.0))
    if 2 in dims:
        spaces.append(VectorSpace(2, 0.0, 1.0))
    if 3 in dims:
        spaces.append(VectorSpace(3, -1.0, 0.0))
    return tuple(spaces)


def validate_algorithm_on_vectors(
        objective: Objective | Callable[[VectorSpace], Objective],
        algorithm: Algorithm | Callable[[VectorSpace, Objective], Algorithm],
        max_fes: int = 100,
        uses_all_fes_if_goal_not_reached=True,
        dims: Iterable[int] = DIMENSIONS_FOR_TESTS) -> None:
    """
    Check the validity of a black-box algorithm on vector problems.

    :param algorithm: the algorithm or algorithm factory
    :param objective: the objective function or function factory
    :param max_fes: the maximum number of FEs
    :param uses_all_fes_if_goal_not_reached: will the algorithm use all FEs
        unless it reaches the goal?
    :param dims: the dimensions
    """
    if not (isinstance(algorithm, Algorithm) or callable(algorithm)):
        raise type_error(algorithm, "algorithm", Algorithm, True)
    if not (isinstance(objective, Objective) or callable(objective)):
        raise type_error(objective, "objective", Objective, True)
    if not isinstance(dims, Iterable):
        raise type_error(dims, "dims", Iterable)

    for space in vectors_for_tests(dims):
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


def make_vector_valid(space: VectorSpace) -> \
        Callable[[Generator, np.ndarray], np.ndarray]:
    """
    Create a function that can make a vector space element valid.

    :param space: the vector space
    :returns: the function
    """

    def __make_valid(prnd: Generator,
                     x: np.ndarray,
                     ppp=space) -> np.ndarray:
        np.copyto(x, prnd.uniform(ppp.lower_bound,
                                  ppp.upper_bound, ppp.dimension))
        return x

    return __make_valid


def validate_algorithm_on_ackley(
        algorithm: Algorithm | Callable[[VectorSpace, Objective], Algorithm],
        uses_all_fes_if_goal_not_reached: bool = True,
        dims: Iterable[int] = DIMENSIONS_FOR_TESTS) -> None:
    """
    Check the validity of a black-box algorithm on Ackley's function.

    :param algorithm: the algorithm or algorithm factory
    :param uses_all_fes_if_goal_not_reached: will the algorithm use all FEs
        unless it reaches the goal?
    :param dims: the dimensions
    """
    validate_algorithm_on_vectors(
        Ackley(), algorithm,
        uses_all_fes_if_goal_not_reached=uses_all_fes_if_goal_not_reached,
        dims=dims)


def validate_op0_on_1_vectors(
        op0: Op0 | Callable[[VectorSpace], Op0],
        search_space: VectorSpace,
        number_of_samples: int | None = None,
        min_unique_samples:
        int | Callable[[int, VectorSpace], int] | None
        = lambda i, _: max(1, i // 3)) -> None:
    """
    Validate the nullary operator on one `VectorSpace` instance.

    :param op0: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: dict[str, Any] = {
        "op0": op0(search_space) if callable(op0) else op0,
        "search_space": search_space,
        "make_search_space_element_valid":
            make_vector_valid(search_space)
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op0(**args)


def validate_op0_on_vectors(
        op0: Op0 | Callable[[VectorSpace], Op0],
        number_of_samples: int | None = None,
        min_unique_samples:
        int | Callable[[int, VectorSpace], int] | None
        = lambda i, _: max(1, i // 3)) -> None:
    """
    Validate the nullary operator on default `VectorSpace` instance.

    :param op0: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    for vs in vectors_for_tests():
        validate_op0_on_1_vectors(
            op0, vs, number_of_samples, min_unique_samples)
