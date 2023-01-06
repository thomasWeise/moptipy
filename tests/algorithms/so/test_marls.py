"""Test the Memetic Algorithm with hard-coded RLS."""

from numpy.random import Generator, default_rng

from moptipy.algorithms.so.marls import MARLS
from moptipy.api.objective import Objective
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.operators.permutations.op2_gap import (
    Op2GeneralizedAlternatingPosition,
)
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import validate_algorithm_on_onemax
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_marls_on_jssp_random() -> None:
    """Validate the ma-rls on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> MARLS:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        return MARLS(Op0Shuffle(search_space),
                     Op1Swap2(),
                     Op2GeneralizedAlternatingPosition(search_space),
                     int(random.integers(2, 12)), int(random.integers(1, 12)),
                     int(random.integers(2, 32)))

    validate_algorithm_on_jssp(create)


def test_marls_on_onemax_random() -> None:
    """Validate the ma-rls on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> MARLS:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        return MARLS(Op0Random(), Op1Flip1(), Op2Uniform(),
                     int(random.integers(2, 12)), int(random.integers(1, 12)),
                     int(random.integers(2, 32)))

    validate_algorithm_on_onemax(create)
