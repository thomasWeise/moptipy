"""Test the Nelder-Mead Downhill Simplex."""

from moptipy.algorithms.so.vector.nelder_mead import NelderMead
from moptipy.operators.vectors.op0_uniform import Op0Uniform
from moptipy.spaces.bounded_vectorspace import BoundedVectorSpace
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.tests.on_vectors import validate_algorithm_on_ackley


def test_nelder_mead_on_ackley():
    """Validate the Nelder-Mead algorithm on Ackley's Function."""

    def create(space: VectorSpace, _):
        if isinstance(space, BoundedVectorSpace):
            mi = space.x_min
            ma = space.x_max
            return NelderMead(Op0Uniform(mi, ma), mi, ma)
        return NelderMead(Op0Uniform(-100.0, 100.0))

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)