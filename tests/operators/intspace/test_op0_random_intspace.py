"""Test the nullary random integers creation operation."""
from moptipy.operators.intspace.op0_random import Op0Random
from moptipy.spaces.intspace import IntSpace
from moptipy.tests.on_intspaces import validate_op0_on_intspaces


def test_op0_random_intspace() -> None:
    """Test the nullary random intspace creation operation."""

    def _min_diff(samples: int, bss: IntSpace) -> int:
        return max(1, min(samples, bss.dimension) // 2)

    validate_op0_on_intspaces(Op0Random, min_unique_samples=_min_diff)
