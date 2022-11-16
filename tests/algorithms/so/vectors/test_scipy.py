"""Test the algorithms wrapped around the SciPy API."""

from math import inf

from moptipy.algorithms.so.vector.scipy import NelderMead, Powell, BGFS, CG, \
    SLSQP, TNC
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


def test_bgfs_on_ackley():
    """Validate BGFS on Ackley's Function."""

    def create(space: VectorSpace, _):
        if isinstance(space, BoundedVectorSpace):
            mi = space.min_value
            ma = space.max_value
            return BGFS(Op0Uniform(mi, ma))
        return BGFS(Op0Uniform(-100.0, 100.0), -inf, inf)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)


def test_cg_on_ackley():
    """Validate CG on Ackley's Function."""

    def create(space: VectorSpace, _):
        if isinstance(space, BoundedVectorSpace):
            mi = space.min_value
            ma = space.max_value
            return CG(Op0Uniform(mi, ma))
        return CG(Op0Uniform(-100.0, 100.0), -inf, inf)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)


def test_slsqp_on_ackley():
    """Validate SLSQP on Ackley's Function."""

    def create(space: VectorSpace, _):
        if isinstance(space, BoundedVectorSpace):
            mi = space.min_value
            ma = space.max_value
            return SLSQP(Op0Uniform(mi, ma))
        return SLSQP(Op0Uniform(-100.0, 100.0), -inf, inf)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)


def test_tnc_on_ackley():
    """Validate TNC on Ackley's Function."""

    def create(space: VectorSpace, _):
        if isinstance(space, BoundedVectorSpace):
            mi = space.min_value
            ma = space.max_value
            return TNC(Op0Uniform(mi, ma))
        return TNC(Op0Uniform(-100.0, 100.0), -inf, inf)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)
