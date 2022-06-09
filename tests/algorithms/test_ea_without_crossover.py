"""Test the ea_no_cr."""
from numpy.random import Generator, default_rng

# noinspection PyPackageRequirements
from moptipy.algorithms.ea_without_crossover import EAnoCR
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import validate_algorithm_on_onemax, \
    validate_algorithm_on_leadingones
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_ea_no_cr_on_jssp():
    """Validate the ea_no_cr on the JSSP."""

    def create(instance: Instance,
               search_space: Permutations):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        random: Generator = default_rng()
        return EAnoCR(Op0Shuffle(search_space), Op1Swap2(),
                      int(random.integers(1, 12)),
                      int(random.integers(1, 12)))

    validate_algorithm_on_jssp(create)


def test_ea_no_cr_on_onemax():
    """Validate the ea_no_cr on the OneMax problem."""

    def create(bs: BitStrings):
        assert isinstance(bs, BitStrings)
        random: Generator = default_rng()
        return EAnoCR(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                      int(random.integers(1, 12)),
                      int(random.integers(1, 12)))

    validate_algorithm_on_onemax(create)


def test_ea_no_cr_on_leadingones():
    """Validate the ea_no_cr on the LeadingOnes problem."""

    def create(bs: BitStrings):
        assert isinstance(bs, BitStrings)
        random: Generator = default_rng()
        return EAnoCR(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                      int(random.integers(1, 12)),
                      int(random.integers(1, 12)))

    validate_algorithm_on_leadingones(create)
