"""Test the RLS."""
from typing import Final

from numpy.random import default_rng, Generator

from moptipy.algorithms.so.ea import EA
from moptipy.algorithms.so.fea1plus1 import FEA1plus1
from moptipy.algorithms.so.fitness_ea import FitnessEA
from moptipy.algorithms.so.fitnesses.ffa import FFA
from moptipy.algorithms.so.rls import RLS
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.tests.on_bitstrings import verify_algorithms_equivalent


def test_opoea_equals_rls():
    """Test whether the (1+1)-EA performs exactly as RLS."""
    op0: Final[Op0] = Op0Random()
    op1: Final[Op1] = Op1Flip1()
    op2: Final[Op2] = Op2Uniform()

    verify_algorithms_equivalent([
        lambda bs, f: RLS(op0, op1),
        lambda bs, f: EA(op0, op1, op2, 1, 1, 0.0),
        lambda bs, f: FitnessEA(op0, op1, op2, 1, 1, 0.0),
    ])


def test_fitness_ea_equals_ea():
    """Ensure that the EAs with and without fitness are identical."""
    op0: Final[Op0] = Op0Random()
    op1: Final[Op1] = Op1Flip1()
    op2: Final[Op2] = Op2Uniform()
    random: Final[Generator] = default_rng()
    mu: Final[int] = int(random.integers(2, 10))
    lambda_: Final[int] = int(random.integers(1, 10))
    br: Final[float] = float(random.uniform(0.1, 0.9))

    verify_algorithms_equivalent([
        lambda bs, f: EA(op0, op1, op2, mu, lambda_, br),
        lambda bs, f: FitnessEA(op0, op1, op2, mu, lambda_, br),
    ])


def test_fitness_ea_with_ffa_equals_fea():
    """Ensure that the FEA and the EA with FFA are identical."""
    op0: Final[Op0] = Op0Random()
    op1: Final[Op1] = Op1Flip1()

    verify_algorithms_equivalent([
        lambda bs, f: FEA1plus1(op0, op1),
        lambda bs, f: FitnessEA(op0, op1, fitness=FFA(f)),
    ])
