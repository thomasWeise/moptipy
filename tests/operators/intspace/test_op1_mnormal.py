"""Test the unary m over n flips operation."""
from moptipy.operators.intspace.op1_mnormal import Op1MNormal
from moptipy.spaces.intspace import IntSpace
from moptipy.tests.on_intspaces import (
    validate_op1_on_intspaces,
)


def test_op1_mnormal() -> None:
    """Test the unary bit flip operation."""
    for flip_1 in [True, False]:
        for sd in [1.0, 2.0, 3.0]:
            for m in [1, 2, 3, 10, 20]:
                def __op1(space: IntSpace, _f=flip_1, _sd=sd, _m=m) \
                        -> Op1MNormal:
                    return Op1MNormal(space, _m, _f, _sd)

                def __check(space: IntSpace, _m=m, _s=sd) -> bool:
                    return (space.dimension >= _m) and _s <= (
                        space.max_value - space.min_value + 1)

                def __min_d_s(s: int, space: IntSpace, _f=flip_1) -> int:
                    return max(1, (2 * min(
                        s, space.dimension)) // 3) if _f else 1

                validate_op1_on_intspaces(__op1, min_unique_samples=__min_d_s,
                                          space_filter=__check)
