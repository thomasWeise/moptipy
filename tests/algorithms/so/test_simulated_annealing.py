"""Test the Simulated Annealing."""

from numpy.random import Generator, default_rng

from moptipy.algorithms.modules.temperature_schedule import (
    ExponentialSchedule,
    LogarithmicSchedule,
)
from moptipy.algorithms.so.simulated_annealing import SimulatedAnnealing
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


def test_sa_on_jssp_random() -> None:
    """Validate the simulated annealing on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> SimulatedAnnealing:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        ts = ExponentialSchedule if random.integers(2) <= 0 \
            else LogarithmicSchedule
        return SimulatedAnnealing(
            Op0Shuffle(search_space), Op1Swap2(), ts(
                random.uniform(1, 1000), random.uniform(0.1, 0.9)))
    validate_algorithm_on_jssp(create)


def test_sa_on_onemax_random() -> None:
    """Validate the simulated annealing on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> SimulatedAnnealing:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        ts = ExponentialSchedule if random.integers(2) <= 0 \
            else LogarithmicSchedule
        return SimulatedAnnealing(
            Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
            ts(random.uniform(1, 1000), random.uniform(0.1, 0.9)))
    validate_algorithm_on_onemax(create)


def test_sa_on_leadingones() -> None:
    """Validate the ea on the LeadingOnes problem."""

    def create(bs: BitStrings, objective: Objective) -> SimulatedAnnealing:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        ts = ExponentialSchedule if random.integers(2) <= 0 \
            else LogarithmicSchedule
        return SimulatedAnnealing(
            Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
            ts(random.uniform(1, 1000), random.uniform(0.1, 0.9)))
    validate_algorithm_on_leadingones(create)
