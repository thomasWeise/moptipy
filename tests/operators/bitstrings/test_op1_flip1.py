"""Test the unary 1 flip operation."""
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.tests.on_bitstrings import (
    bitstrings_for_tests,
    validate_op1_on_1_bitstrings,
)


def test_op1_1flip() -> None:
    """Test the unary bit flip operation."""
    op: Op1Flip1 = Op1Flip1()

    for bs in bitstrings_for_tests():
        validate_op1_on_1_bitstrings(
            op1=op, search_space=bs,
            min_unique_samples=lambda n, bss:
                max(1, min(n, bss.dimension) // 4))
