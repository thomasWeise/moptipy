"""Test random sampling."""
# noinspection PyPackageRequirements
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.tests.on_bitstrings import validate_algorithm_on_onemax, \
    validate_algorithm_on_leadingones
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_random_sampling_on_jssp():
    """Validate random sampling on the JSSP."""

    def create(instance: Instance,
               search_space: PermutationsWithRepetitions):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, PermutationsWithRepetitions)
        return RandomSampling(Op0Shuffle(search_space))

    validate_algorithm_on_jssp(algorithm=create)


def test_random_sampling_on_onemax():
    """Validate the random sampling on the onemax problmem."""

    def create(bs: BitStrings):
        assert isinstance(bs, BitStrings)
        return RandomSampling(Op0Random())

    validate_algorithm_on_onemax(create)


def test_random_sampling_on_leadingones():
    """Validate the random sampling on the leadingones problmem."""

    def create(bs: BitStrings):
        assert isinstance(bs, BitStrings)
        return RandomSampling(Op0Random())

    validate_algorithm_on_leadingones(create)
