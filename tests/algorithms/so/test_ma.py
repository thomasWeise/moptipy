"""Test the Memetic Algorithm."""

from numpy.random import Generator, default_rng

from moptipy.algorithms.so.ma import MA
from moptipy.algorithms.so.rls import RLS
from moptipy.api.objective import Objective
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.operators.op0_forward import Op0Forward
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.operators.permutations.op2_gap import (
    Op2GeneralizedAlternatingPosition,
)
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import validate_algorithm_on_onemax
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_ma_on_jssp_random() -> None:
    """Validate the ma on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> MA:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        return MA(Op0Shuffle(search_space),
                  Op2GeneralizedAlternatingPosition(search_space),
                  RLS(Op0Forward(), Op1Swap2()),
                  int(random.integers(2, 12)), int(random.integers(1, 12)),
                  int(random.integers(2, 32)))

    validate_algorithm_on_jssp(create)


def test_ma_on_onemax_random() -> None:
    """Validate the ma on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> MA:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        return MA(Op0Random(), Op2Uniform(), RLS(Op0Forward(), Op1Flip1()),
                  int(random.integers(2, 12)), int(random.integers(1, 12)),
                  int(random.integers(2, 32)))

    validate_algorithm_on_onemax(create)
