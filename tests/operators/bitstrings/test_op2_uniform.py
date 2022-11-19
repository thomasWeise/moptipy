"""Test the uniform crossover operation."""

from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.on_bitstrings import validate_op2_on_bitstrings


def test_op2_uniform() -> None:
    """Test the uniform crossover operation."""

    def _min_diff(samples: int, bss: BitStrings) -> int:
        return max(1, int(min(samples, bss.dimension) ** 0.3))

    validate_op2_on_bitstrings(Op2Uniform(),
                               min_unique_samples=_min_diff)
