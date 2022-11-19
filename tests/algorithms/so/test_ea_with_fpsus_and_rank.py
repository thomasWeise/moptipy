"""Test the General Evolutionary Algorithm with Roulette Wheel Selection."""
from typing import Final

from numpy.random import Generator, default_rng

from moptipy.algorithms.modules.selections.fitness_proportionate_sus import (
    FitnessProportionateSUS,
)
from moptipy.algorithms.so.fitnesses.direct import Direct
from moptipy.algorithms.so.fitnesses.rank import Rank
from moptipy.algorithms.so.general_ea import GeneralEA
from moptipy.api.objective import Objective
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
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


def test_general_ea_on_jssp_random() -> None:
    """Validate the ea on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> GeneralEA:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        mu: Final[int] = int(random.integers(1, 12))
        lambda_: Final[int] = int(random.integers(1, 12))
        return GeneralEA(Op0Shuffle(search_space), Op1Swap2(),
                         Op2GeneralizedAlternatingPosition(search_space),
                         mu, lambda_,
                         0.0 if mu <= 1 else float(random.random()),
                         Rank(),
                         FitnessProportionateSUS(1.0 / (mu + lambda_ + 5)))

    validate_algorithm_on_jssp(create)


def test_general_ea_on_onemax_random() -> None:
    """Validate the ea on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> GeneralEA:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        mu: Final[int] = int(random.integers(1, 12))
        lambda_: Final[int] = int(random.integers(1, 12))
        return GeneralEA(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                         Op2Uniform(), mu, lambda_,
                         0.0 if mu <= 1 else float(random.random()),
                         Direct(),
                         FitnessProportionateSUS())

    validate_algorithm_on_onemax(create)
