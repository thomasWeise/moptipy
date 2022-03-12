"""Test the nullary bitstrings operation."""

from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.on_bitstrings import validate_op0_on_bitstrings


def test_op0_shuffle():
    """Test the nullary bitstrings operation."""

    def _min_diff(samples: int, bss: BitStrings) -> int:
        return max(1, min(samples, bss.dimension) // 2)

    validate_op0_on_bitstrings(Op0Random(),
                               min_unique_samples=_min_diff)
