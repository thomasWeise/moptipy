"""Test the uniform crossover operation."""

from moptipy.operators.intspace.op2_uniform import Op2Uniform
from moptipy.spaces.intspace import IntSpace
from moptipy.tests.on_intspaces import validate_op2_on_intspaces


def test_op2_uniform_intspace() -> None:
    """Test the uniform crossover operation."""

    def _min_diff(samples: int, bss: IntSpace) -> int:
        return max(1, int(min(samples, bss.dimension) ** 0.3))

    validate_op2_on_intspaces(Op2Uniform(),
                              min_unique_samples=_min_diff)
