"""Test the unary swap-exactly-n operation."""
from collections.abc import Iterable
from typing import Callable, Final

import numpy as np
from numpy.random import Generator, default_rng
from pycommons.types import check_int_range, type_error

from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap_exactly_n import (
    Op1SwapExactlyN,
    apply_move,
    find_move,
    get_max_changes,
)
from moptipy.operators.tools import (
    exponential_step_size,
    inv_exponential_step_size,
)
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_permutations import (
    permutations_for_tests,
    validate_op1_with_step_size_on_permutations,
)


def test_operator_components() -> None:
    """Test the `find_move` function."""
    random = default_rng()
    ri = random.integers
    permute = random.permutation
    for perm in permutations_for_tests(lambda p: p.dimension < 50):
        n = check_int_range(perm.n(), "perm.dimension", 1, 1_000_000)
        mc = check_int_range(get_max_changes(perm.blueprint),
                             "get_max_changes", max(n, 2), perm.dimension)
        is_plain_perm: bool = perm.dimension == perm.n()
        if is_plain_perm and (mc != perm.dimension):
            raise ValueError(
                f"{perm} is a normal permutation, get_max_changes must "
                f"equal dimension {perm.dimension}, but is {mc}")

        indices = np.array(range(perm.dimension), int)
        temp = np.empty(perm.dimension, int)
        dest = perm.create()
        for ss in range(2, min(50, mc + 1)):
            x = permute(perm.blueprint)
            steps: int = int(1 if ri(0, 2) < 1 else ri(1, 1000))
            mv = find_move(x, indices, ss, random, steps, temp)
            if not isinstance(mv, np.ndarray):
                raise type_error(
                    mv,
                    f"find_move({x}, 0:{len(x) - 1}, {ss}, ri, {steps}, ...)",
                    np.ndarray)
            lmv = len(mv)
            if not (2 <= lmv <= ss):
                raise ValueError(
                    f"invalid length {len(mv)} of returned move {mv} for"
                    f" {perm} with x={x} and max_trials={steps}, must be in "
                    f"[2,{ss}].")
            if is_plain_perm and (lmv != ss):
                raise ValueError(
                    f"On the plain permutation {perm}, a step-size request "
                    f"of {ss} must result in a move of length {ss} but "
                    f"resulted in one of length {lmv}.")

            lu = len(set(mv))
            if lu != lmv:
                raise ValueError(f"moves contain duplicate value: {mv}")
            orig = x[mv]
            for roll in [-1, 1]:
                mod = np.roll(orig, roll)
                cs = sum(mod != orig)
                if cs != lmv:
                    raise ValueError(
                        f"invalid move {mv} of length {lmv} copies {orig} to"
                        f" {mod} for roll({roll}), which has {cs} "
                        f"differences instead of {lmv}.")

            xcpy = x.copy()
            apply_move(x, dest, mv, random, 0 if (ri(3) <= 0) else ri(1, 1000))
            if not perm.is_equal(xcpy, x):
                raise ValueError("applying move destroyed source")
            perm.validate(dest)
            dif = sum(x != dest)
            if dif != lmv:
                raise ValueError(
                    f"applying move {mv} changed {dif} loci, "
                    f"but should change {lmv} from {x} to {dest}")


def test_op1_swapxn() -> None:
    """Test the unary swap-exactly-n operation."""
    def _min_unique(samples, pwrx) -> int:
        return max(1, min(samples, pwrx.n()) // 2)

    def _get_step_size(perm, a: np.ndarray, b: np.ndarray) -> float | None:
        if perm.n() != perm.dimension:
            return None
        return inv_exponential_step_size(check_int_range(
            int(sum(a != b)), "sum(a!=b)", 2, perm.dimension),
            2, get_max_changes(perm.blueprint))

    def _get_step_sizes(p: Permutations,
                        ri=default_rng().integers) -> Iterable[float]:
        maxss = get_max_changes(p.blueprint)
        if maxss > 2:
            res = {2, maxss}
            if maxss > 3:
                res.add(int(ri(3, maxss)))
            else:
                res.add(3)
            return list({inv_exponential_step_size(j, 2, maxss) for j in res
                         if 2 <= j <= maxss})
        return [1.0]

    validate_op1_with_step_size_on_permutations(
        Op1SwapExactlyN, None, _min_unique,
        _get_step_sizes, _get_step_size,
        lambda p: (p.dimension < 50) and (get_max_changes(p.blueprint) >= 2))


def test_op1_swapxn_exact() -> None:
    """Test the exact number of swaps by a swap-exactly-n op."""
    random: Final[Generator] = default_rng()
    perm: Final[Permutations] = Permutations.standard(
        int(random.integers(10, 100)))
    x1: Final[np.ndarray] = perm.create()
    Op0Shuffle(perm).op0(random, x1)
    x2: Final[np.ndarray] = perm.create()
    op1: Final[Op1SwapExactlyN] = Op1SwapExactlyN(perm)
    op1.initialize()
    op: Final[Callable[[Generator, np.ndarray,
                        np.ndarray, float], None]] = op1.op1

    op(random, x2, x1, 0.0)
    assert sum(x1 != x2) == 2
    op(random, x2, x1, 1.0)
    assert sum(x1 != x2) == len(x2)

    for _ in range(1000):
        steps = int(random.integers(0, 10001)) / 10000
        assert 0.0 <= steps <= 1.0
        changes = exponential_step_size(steps, 2, len(x1))
        assert 2 <= changes <= len(x1)
        op(random, x2, x1, steps)
        assert sum(x1 != x2) == changes

    for changes in range(2, len(x2) + 1):
        steps = inv_exponential_step_size(changes, 2, len(x2))
        assert 0.0 <= steps <= 1.0
        op(random, x2, x1, steps)
        assert sum(x1 != x2) == changes
