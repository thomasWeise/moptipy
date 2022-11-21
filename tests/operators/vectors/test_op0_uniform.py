"""Test the nullary uniform operation."""
from moptipy.operators.vectors.op0_uniform import Op0Uniform
from moptipy.tests.on_vectors import validate_op0_on_vectors


def test_op0_uniform() -> None:
    """Test the nullary uniform sampling operation."""
    validate_op0_on_vectors(Op0Uniform)
