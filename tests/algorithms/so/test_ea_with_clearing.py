"""Test the Evolutionary Algorithm with clearing."""
from typing import Final

from numpy.random import Generator, default_rng

from moptipy.algorithms.so.ea_with_clearing import EAwithClearing
from moptipy.api.objective import Objective
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.operators.permutations.op2_gap import \
    Op2GeneralizedAlternatingPosition
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import validate_algorithm_on_onemax, \
    validate_algorithm_on_leadingones
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_eac_on_jssp_random():
    """Validate the ea on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        mu: Final[int] = int(random.integers(1, 12))
        return EAwithClearing(Op0Shuffle(search_space), Op1Swap2(),
                              Op2GeneralizedAlternatingPosition(search_space),
                              mu, int(random.integers(1, 12)),
                              0.0 if mu <= 1 else float(random.random()))

    validate_algorithm_on_jssp(create)


def test_ea_on_onemax_random():
    """Validate the ea on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective):
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        mu: Final[int] = int(random.integers(1, 12))
        return EAwithClearing(
            Op0Random(), Op1MoverNflip(bs.dimension, 1, True), Op2Uniform(),
            mu, int(random.integers(1, 12)),
            0.0 if mu <= 1 else float(random.random()))

    validate_algorithm_on_onemax(create)


def test_eac_on_leadingones():
    """Validate the ea on the LeadingOnes problem."""

    def create(bs: BitStrings, objective: Objective):
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        mu: Final[int] = int(random.integers(1, 12))
        return EAwithClearing(
            Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
            Op2Uniform(), mu, int(random.integers(1, 12)),
            0.0 if mu <= 1 else float(random.random()))

    validate_algorithm_on_leadingones(create)
