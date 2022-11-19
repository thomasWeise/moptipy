"""Test random walks."""
from moptipy.algorithms.random_walk import RandomWalk
from moptipy.api.objective import Objective
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import (
    validate_algorithm_on_leadingones,
    validate_algorithm_on_onemax,
)
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_random_walk_on_jssp() -> None:
    """Validate a random walk on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> RandomWalk:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        return RandomWalk(Op0Shuffle(search_space), Op1Swap2())

    validate_algorithm_on_jssp(algorithm=create)


def test_random_walk_on_onemax() -> None:
    """Validate the random walk on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> RandomWalk:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return RandomWalk(Op0Random(), Op1MoverNflip(bs.dimension, 1, True))

    validate_algorithm_on_onemax(create)


def test_random_walk_on_leadingones() -> None:
    """Validate the random walk on the LeadingOnes problem."""

    def create(bs: BitStrings, objective: Objective) -> RandomWalk:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return RandomWalk(Op0Random(), Op1MoverNflip(bs.dimension, 1, True))

    validate_algorithm_on_leadingones(create)
