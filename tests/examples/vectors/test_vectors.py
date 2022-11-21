"""Test the several vector objective function."""
from math import inf, nextafter
from typing import Final

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.examples.vectors.ackley import Ackley
from moptipy.examples.vectors.sphere import Sphere
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.tests.objective import validate_objective


def _sphere(x: np.ndarray, ub: float) -> float:
    """The comparison implementation of sphere."""
    s: float = 0.0
    for xx in x:
        if xx:
            s += float(xx) * float(xx)
    if (s < 0.0) or (s > ub):
        raise ValueError(f"x={x}, len(x)={len(x)}, sphere(x)={s}, ub={ub}!")
    return s


def random_vector(random: Generator, x: np.ndarray,
                  min_value: float, max_value: float) -> np.ndarray:
    """
    Create a random vector.

    :param random: the random number generator
    :param x: the destination and result array
    :param min_value: the minimal x value
    :param max_value: the maximal x value
    :returns: the array `x`
    """
    for i in range(len(x)):
        x[i] = random.uniform(min_value, max_value)
    return x


def test_sphere() -> None:
    """Test the sphere objective function."""
    random: Final[Generator] = default_rng()
    space: VectorSpace = VectorSpace(
        dimension=int(random.integers(2, 32)),
        lower_bound=float(-1 - (random.random() * 100)),
        upper_bound=float(1 + (random.random() * 100)))
    f: Sphere = Sphere()

    ub = nextafter((max(abs(space.lower_bound[0]),
                        abs(space.upper_bound[0])) ** 2)
                   * space.dimension, inf)

    validate_objective(
        objective=f,
        solution_space=space,
        make_solution_space_element_valid=lambda
        r, xx, xmi=float(space.lower_bound[0]),
        xma=float(space.upper_bound[0]):
        random_vector(r, xx, xmi, xma),
        is_deterministic=True,
        lower_bound_threshold=0,
        upper_bound_threshold=inf,
        must_be_equal_to=lambda x, uub=ub: _sphere(x, uub))


def test_ackley() -> None:
    """Test the sphere objective function."""
    random: Final[Generator] = default_rng()
    space: VectorSpace = VectorSpace(int(random.integers(2, 32)),
                                     float(-1 - (random.random() * 100)),
                                     float(1 + (random.random() * 100)))
    f: Ackley = Ackley()

    validate_objective(
        objective=f,
        solution_space=space,
        make_solution_space_element_valid=lambda
        r, xx, xmi=float(space.lower_bound[0]),
        xma=float(space.upper_bound[0]):
        random_vector(r, xx, xmi, xma),
        is_deterministic=True,
        lower_bound_threshold=0,
        upper_bound_threshold=inf)
