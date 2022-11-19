"""Test the unary m over n flips operation."""
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.on_bitstrings import (
    bitstrings_for_tests,
    validate_op1_on_1_bitstrings,
)


def test_op1_m_over_n_flips() -> None:
    """Test the unary bit flip operation."""
    for flip_1 in [True, False]:
        for bs in bitstrings_for_tests():
            for m in range(1, 1 + min(bs.dimension, 5)):
                def _min_diff(samples: int, bss: BitStrings) -> int:
                    return max(1, min(samples, bss.dimension) // 3)

                validate_op1_on_1_bitstrings(
                    op1=Op1MoverNflip(bs.dimension, m, flip_1),
                    search_space=bs,
                    min_unique_samples=_min_diff)
