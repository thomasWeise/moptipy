"""Test the Evolutionary Algorithm."""
from typing import Final

from numpy.random import Generator, default_rng

from moptipy.algorithms.so.ea import EA
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
from moptipy.tests.on_bitstrings import (
    validate_algorithm_on_leadingones,
    validate_algorithm_on_onemax,
)
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_ea_on_jssp_random() -> None:
    """Validate the ea on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> EA:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        mu: Final[int] = int(random.integers(1, 12))
        return EA(Op0Shuffle(search_space), Op1Swap2(),
                  Op2GeneralizedAlternatingPosition(search_space),
                  mu, int(random.integers(1, 12)),
                  0.0 if mu <= 1 else float(random.random()))

    validate_algorithm_on_jssp(create)


def test_ea_on_jssp_1_1_0() -> None:
    """Validate the ea using only mutation on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> EA:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        return EA(Op0Shuffle(search_space), Op1Swap2(),
                  Op2GeneralizedAlternatingPosition(search_space),
                  1, 1, 0.0)

    validate_algorithm_on_jssp(create)


def test_ea_on_jssp_10_10_03() -> None:
    """Validate the ea using crossover and mutation on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> EA:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        return EA(Op0Shuffle(search_space), Op1Swap2(),
                  Op2GeneralizedAlternatingPosition(search_space),
                  10, 10, 0.3)

    validate_algorithm_on_jssp(create)


def test_ea_on_jssp_10_10_1() -> None:
    """Validate the ea using only crossover on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> EA:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        return EA(Op0Shuffle(search_space), Op1Swap2(),
                  Op2GeneralizedAlternatingPosition(search_space),
                  10, 10, 1.0)

    validate_algorithm_on_jssp(create)


def test_ea_on_onemax_random() -> None:
    """Validate the ea on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> EA:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        mu: Final[int] = int(random.integers(1, 12))
        return EA(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                  Op2Uniform(),
                  mu, int(random.integers(1, 12)),
                  0.0 if mu <= 1 else float(random.random()))

    validate_algorithm_on_onemax(create)


def test_ea_on_onemax_1_1_0() -> None:
    """Validate the ea on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> EA:
        assert isinstance(objective, Objective)
        assert isinstance(bs, BitStrings)
        return EA(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                  Op2Uniform(), 1, 1, 0.0)

    validate_algorithm_on_onemax(create)


def test_ea_on_onemax_10_10_03() -> None:
    """Validate the ea on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> EA:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return EA(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                  Op2Uniform(), 10, 10, 0.3)

    validate_algorithm_on_onemax(create)


def test_ea_on_onemax_10_10_1() -> None:
    """Validate the ea on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> EA:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return EA(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                  Op2Uniform(), 10, 10, 1.0)

    validate_algorithm_on_onemax(create)


def test_ea_on_leadingones() -> None:
    """Validate the ea on the LeadingOnes problem."""

    def create(bs: BitStrings, objective: Objective) -> EA:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        mu: Final[int] = int(random.integers(1, 12))
        return EA(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                  Op2Uniform(),
                  mu, int(random.integers(1, 12)),
                  0.0 if mu <= 1 else float(random.random()))

    validate_algorithm_on_leadingones(create)


def test_ea_naming() -> None:
    """Test the naming convention of the EA."""
    op0: Final[Op0Random] = Op0Random()
    op1: Final[Op1MoverNflip] = Op1MoverNflip(10, 1, True)
    n1: Final[str] = str(op1)
    op2: Final[Op2Uniform] = Op2Uniform()
    n2: Final[str] = str(op2)

    ea: EA = EA(op0, op1, op2, 10, 5, 0.5)
    assert str(ea) == f"ea_10_5_0d5_{n2}_{n1}"

    ea = EA(op0, op1, op2, 10, 5, 0.0)
    assert str(ea) == f"ea_10_5_{n1}"

    ea = EA(op0, op1, None, 10, 5, 0.0)
    assert str(ea) == f"ea_10_5_{n1}"

    ea = EA(op0, op1, None, 10, 5, None)
    assert str(ea) == f"ea_10_5_{n1}"

    ea = EA(op0, op1, op2, 10, 5, 1.0)
    assert str(ea) == f"ea_10_5_{n2}"

    ea = EA(op0, None, op2, 10, 5, 1.0)
    assert str(ea) == f"ea_10_5_{n2}"

    ea = EA(op0, None, op2, 10, 5, None)
    assert str(ea) == f"ea_10_5_{n2}"
