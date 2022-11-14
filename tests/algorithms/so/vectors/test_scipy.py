"""Test the Nelder-Mead Downhill Simplex."""

from math import inf

from moptipy.algorithms.so.vector.scipy import NelderMead, Powell
from moptipy.operators.vectors.op0_uniform import Op0Uniform
from moptipy.spaces.bounded_vectorspace import BoundedVectorSpace
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.tests.on_vectors import validate_algorithm_on_ackley


def test_nelder_mead_on_ackley():
    """Validate the Nelder-Mead algorithm on Ackley's Function."""

    def create(space: VectorSpace, _):
        if isinstance(space, BoundedVectorSpace):
            mi = space.min_value
            ma = space.max_value
            return NelderMead(Op0Uniform(mi, ma))
        return NelderMead(Op0Uniform(-100.0, 100.0), -inf, inf)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)


def test_powell_on_ackley():
    """Validate Powell's algorithm on Ackley's Function."""

    def create(space: VectorSpace, _):
        if isinstance(space, BoundedVectorSpace):
            mi = space.min_value
            ma = space.max_value
            return Powell(Op0Uniform(mi, ma))
        return Powell(Op0Uniform(-100.0, 100.0), -inf, inf)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)
