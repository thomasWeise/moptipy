"""Test random sampling."""
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.api.objective import Objective
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import (
    validate_algorithm_on_leadingones,
    validate_algorithm_on_onemax,
)
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_random_sampling_on_jssp() -> None:
    """Validate random sampling on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> RandomSampling:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        return RandomSampling(Op0Shuffle(search_space))

    validate_algorithm_on_jssp(algorithm=create)


def test_random_sampling_on_onemax() -> None:
    """Validate the random sampling on the OneMax Problem."""

    def create(bs: BitStrings, objective: Objective) -> RandomSampling:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return RandomSampling(Op0Random())

    validate_algorithm_on_onemax(create)


def test_random_sampling_on_leadingones() -> None:
    """Validate the random sampling on the LeadingOnes problem."""

    def create(bs: BitStrings, objective: Objective) -> RandomSampling:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return RandomSampling(Op0Random())

    validate_algorithm_on_leadingones(create)
