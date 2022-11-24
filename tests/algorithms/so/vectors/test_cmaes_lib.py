"""Test the algorithms wrapped around the `cmaes` API."""

from moptipy.algorithms.so.vector.cmaes_lib import CMAES, BiPopCMAES, SepCMAES
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.tests.on_vectors import (
    DIMENSIONS_FOR_TESTS,
    validate_algorithm_on_ackley,
)


def test_cmaes_on_ackley() -> None:
    """Validate CMAES on Ackley's Function."""

    def create(space: VectorSpace, _) -> CMAES:
        return CMAES(space)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False,
        dims=filter(lambda i: i > 1, DIMENSIONS_FOR_TESTS))


def test_sepcmaes_on_ackley() -> None:
    """Validate Sep-CMAES on Ackley's Function."""

    def create(space: VectorSpace, _) -> SepCMAES:
        return SepCMAES(space)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False,
        dims=filter(lambda i: i > 1, DIMENSIONS_FOR_TESTS))


def test_bipopmaes_on_ackley() -> None:
    """Validate Bi-Pop-CMAES on Ackley's Function."""

    def create(space: VectorSpace, _) -> BiPopCMAES:
        return BiPopCMAES(space)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False,
        dims=filter(lambda i: i > 1, DIMENSIONS_FOR_TESTS))
