"""Test the algorithms wrapped around the PDFO API."""

from moptipy.algorithms.so.vector.pdfo import BOBYQA
from moptipy.operators.vectors.op0_uniform import Op0Uniform
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.tests.on_vectors import validate_algorithm_on_ackley


def test_bobyqa_on_ackley() -> None:
    """Validate BOBYQA on Ackley's Function."""

    def create(space: VectorSpace, _) -> BOBYQA:
        return BOBYQA(Op0Uniform(space), space)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)
