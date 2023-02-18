"""Test the unary bit flip operation with step sizes."""
from typing import Iterable

import numpy as np
from numpy.random import default_rng

from moptipy.operators.bitstrings.op1_flip_m import Op1FlipM
from moptipy.operators.tools import inv_exponential_step_size
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.on_bitstrings import (
    validate_op1_with_step_size_on_bitstrings,
)
from moptipy.utils.types import check_int_range


def test_op1_flip_m() -> None:
    """Test the unary bit flip operation."""
    op: Op1FlipM = Op1FlipM()

    def _get_ss_for_test(b: BitStrings, g=default_rng().integers) \
            -> Iterable[float]:
        bs = {1.0}
        if b.dimension > 1:
            bs.add(0.0)
            if b.dimension > 2:
                bs.update(inv_exponential_step_size(i, 1, b.dimension)
                          for i in g(1, b.dimension, 10))
        return list(bs)

    def _get_ss_from_bits(bs: BitStrings, a: np.ndarray, b: np.ndarray) \
            -> float:
        return inv_exponential_step_size(check_int_range(
            int(sum(a != b)), "sum(a!=b)", 1, bs.dimension),
            1, bs.dimension)

    validate_op1_with_step_size_on_bitstrings(
        op1=op, step_sizes=_get_ss_for_test,
        get_step_size=_get_ss_from_bits,
        min_unique_samples=lambda n, bss: max(
            1, min(n, bss.dimension) // 4))
