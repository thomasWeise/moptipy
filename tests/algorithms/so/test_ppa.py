"""Test the Plant Propagation Algorithm."""

from numpy.random import Generator, default_rng

from moptipy.algorithms.so.ppa import PPA
from moptipy.api.objective import Objective
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip_m import Op1FlipM
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap_exactly_n import Op1SwapExactlyN
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import (
    validate_algorithm_on_onemax,
)
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_ppa_on_jssp_random() -> None:
    """Validate the PPA on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> PPA:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        return PPA(Op0Shuffle(search_space),
                   Op1SwapExactlyN(search_space),
                   int(random.integers(1, 12)),
                   int(random.integers(1, 12)))

    validate_algorithm_on_jssp(create)


def test_ppa_on_jssp_1_1() -> None:
    """Validate the PPA on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> PPA:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        return PPA(Op0Shuffle(search_space),
                   Op1SwapExactlyN(search_space), 1, 1)

    validate_algorithm_on_jssp(create)


def test_ppa_on_onemax_random() -> None:
    """Validate the PPA on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> PPA:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        return PPA(Op0Random(), Op1FlipM(),
                   int(random.integers(1, 12)), int(random.integers(1, 12)))

    validate_algorithm_on_onemax(create)


def test_ppa_on_onemax_1_1() -> None:
    """Validate the PPA on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> PPA:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return PPA(Op0Random(), Op1FlipM(), 1, 1)

    validate_algorithm_on_onemax(create)
