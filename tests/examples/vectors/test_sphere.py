"""Test the sphere objective function."""
from math import nextafter, inf
from typing import Final

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.examples.vectors.sphere import Sphere
from moptipy.spaces.bounded_vectorspace import BoundedVectorSpace
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
                  x_min: float, x_max: float) -> np.ndarray:
    """
    Create a random vector.

    :param random: the random number generator
    :param x: the destination and result array
    :param x_min: the minimal x value
    :param x_max: the maximal x value
    :returns: the array `x`
    """
    for i in range(len(x)):
        x[i] = min(x_max, max(
            x_min, x_min + random.random() * (x_max - x_min)))
    return x


def test_sphere() -> None:
    """Test the sphere objective function."""
    random: Final[Generator] = default_rng()
    space: BoundedVectorSpace = BoundedVectorSpace(
        dimension=int(random.integers(2, 32)),
        x_min=float(-1 - (random.random() * 100)),
        x_max=float(1 + (random.random() * 100)))
    f: Sphere = Sphere()

    ub = nextafter((max(abs(space.x_min), abs(space.x_max)) ** 2)
                   * space.dimension, inf)

    validate_objective(
        objective=f,
        solution_space=space,
        make_solution_space_element_valid=lambda
        r, xx, xmi=space.x_min, xma=space.x_max:
        random_vector(r, xx, xmi, xma),
        is_deterministic=True,
        lower_bound_threshold=0,
        upper_bound_threshold=inf,
        must_be_equal_to=lambda x, uub=ub: _sphere(x, uub))
