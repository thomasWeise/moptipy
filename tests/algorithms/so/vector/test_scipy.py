"""Test the algorithms wrapped around the SciPy API."""

from moptipy.algorithms.so.vector.scipy import (
    BGFS,
    CG,
    DE,
    SLSQP,
    TNC,
    NelderMead,
    Powell,
)
from moptipy.operators.vectors.op0_uniform import Op0Uniform
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.tests.on_vectors import validate_algorithm_on_ackley


def test_nelder_mead_on_ackley() -> None:
    """Validate the Nelder-Mead algorithm on Ackley's Function."""

    def create(space: VectorSpace, _) -> NelderMead:
        return NelderMead(Op0Uniform(space), space)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)


def test_powell_on_ackley() -> None:
    """Validate Powell's algorithm on Ackley's Function."""

    def create(space: VectorSpace, _) -> Powell:
        return Powell(Op0Uniform(space), space)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)


def test_bgfs_on_ackley() -> None:
    """Validate BGFS on Ackley's Function."""

    def create(space: VectorSpace, _) -> BGFS:
        return BGFS(Op0Uniform(space), space)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)


def test_cg_on_ackley() -> None:
    """Validate CG on Ackley's Function."""

    def create(space: VectorSpace, _) -> CG:
        return CG(Op0Uniform(space), space)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)


def test_slsqp_on_ackley() -> None:
    """Validate SLSQP on Ackley's Function."""

    def create(space: VectorSpace, _) -> SLSQP:
        return SLSQP(Op0Uniform(space), space)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)


def test_tnc_on_ackley() -> None:
    """Validate TNC on Ackley's Function."""

    def create(space: VectorSpace, _) -> TNC:
        return TNC(Op0Uniform(space), space)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)


def test_de_on_ackley() -> None:
    """Validate Differential Evolution on Ackley's Function."""

    def create(space: VectorSpace, _) -> DE:
        return DE(space)

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=False)
