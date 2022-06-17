"""Test the ea."""
from typing import Final

from numpy.random import Generator, default_rng

from moptipy.algorithms.ea import EA
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.operators.permutations.op2_sequence import Op2Sequence
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import validate_algorithm_on_onemax, \
    validate_algorithm_on_leadingones
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_ea_on_jssp_random():
    """Validate the ea on the JSSP."""

    def create(instance: Instance,
               search_space: Permutations):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        random: Generator = default_rng()
        mu: Final[int] = int(random.integers(1, 12))
        return EA(Op0Shuffle(search_space), Op1Swap2(),
                  Op2Sequence(search_space),
                  mu, int(random.integers(1, 12)),
                  0.0 if mu <= 1 else float(random.random()))

    validate_algorithm_on_jssp(create)


def test_ea_on_jssp_1_1_0():
    """Validate the ea using only mutation on the JSSP."""

    def create(instance: Instance,
               search_space: Permutations):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        return EA(Op0Shuffle(search_space), Op1Swap2(),
                  Op2Sequence(search_space),
                  1, 1, 0.0)

    validate_algorithm_on_jssp(create)


def test_ea_on_jssp_10_10_03():
    """Validate the ea using crossover and mutation on the JSSP."""

    def create(instance: Instance,
               search_space: Permutations):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        return EA(Op0Shuffle(search_space), Op1Swap2(),
                  Op2Sequence(search_space),
                  10, 10, 0.3)

    validate_algorithm_on_jssp(create)


def test_ea_on_jssp_10_10_1():
    """Validate the ea using only crossover on the JSSP."""

    def create(instance: Instance,
               search_space: Permutations):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        return EA(Op0Shuffle(search_space), Op1Swap2(),
                  Op2Sequence(search_space),
                  10, 10, 1.0)

    validate_algorithm_on_jssp(create)


def test_ea_on_onemax_random():
    """Validate the ea on the OneMax problem."""

    def create(bs: BitStrings):
        assert isinstance(bs, BitStrings)
        random: Generator = default_rng()
        mu: Final[int] = int(random.integers(1, 12))
        return EA(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                  Op2Uniform(),
                  mu, int(random.integers(1, 12)),
                  0.0 if mu <= 1 else float(random.random()))

    validate_algorithm_on_onemax(create)


def test_ea_on_onemax_1_1_0():
    """Validate the ea on the OneMax problem."""

    def create(bs: BitStrings):
        assert isinstance(bs, BitStrings)
        return EA(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                  Op2Uniform(), 1, 1, 0.0)

    validate_algorithm_on_onemax(create)


def test_ea_on_onemax_10_10_03():
    """Validate the ea on the OneMax problem."""

    def create(bs: BitStrings):
        assert isinstance(bs, BitStrings)
        return EA(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                  Op2Uniform(), 10, 10, 0.3)

    validate_algorithm_on_onemax(create)


def test_ea_on_onemax_10_10_1():
    """Validate the ea on the OneMax problem."""

    def create(bs: BitStrings):
        assert isinstance(bs, BitStrings)
        return EA(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                  Op2Uniform(), 10, 10, 1.0)

    validate_algorithm_on_onemax(create)


def test_ea_on_leadingones():
    """Validate the ea on the LeadingOnes problem."""

    def create(bs: BitStrings):
        assert isinstance(bs, BitStrings)
        random: Generator = default_rng()
        mu: Final[int] = int(random.integers(1, 12))
        return EA(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                  Op2Uniform(),
                  mu, int(random.integers(1, 12)),
                  0.0 if mu <= 1 else float(random.random()))

    validate_algorithm_on_leadingones(create)


def test_ea_naming():
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
