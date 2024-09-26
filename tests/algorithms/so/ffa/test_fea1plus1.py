"""Test the (1+1)-FEA."""
from pycommons.io.temp import temp_file

from moptipy.algorithms.so.ffa.fea1plus1 import FEA1plus1
from moptipy.api.execution import Execution
from moptipy.api.objective import Objective
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
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


def __lb() -> int:
    """A mock lower bound."""
    return -1_000_000_000_000_000


def __ub() -> int:
    """A mock upper bound."""
    return 1_000_000_000_000_000


def test_fea1plus1_on_jssp() -> None:
    """Validate the (1+1)-FEA on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> FEA1plus1:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        return FEA1plus1(Op0Shuffle(search_space), Op1Swap2())

    validate_algorithm_on_jssp(create)


def test_fea1plus1_on_onemax() -> None:
    """Validate the (1+1)-FEA on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> FEA1plus1:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return FEA1plus1(Op0Random(), Op1MoverNflip(bs.dimension, 1, True))

    validate_algorithm_on_onemax(create)


def test_fea1plus1_on_onemax_with_large_range() -> None:
    """Validate the (1+1)-FEA on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> FEA1plus1:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        objective.lower_bound = __lb  # type: ignore
        objective.upper_bound = __ub  # type: ignore
        return FEA1plus1(Op0Random(), Op1MoverNflip(bs.dimension, 1, True))

    validate_algorithm_on_onemax(create)


def test_fea1plus1_on_leadingones() -> None:
    """Validate the (1+1)-FEA on the LeadingOnes problem."""

    def create(bs: BitStrings, objective: Objective) -> FEA1plus1:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return FEA1plus1(Op0Random(), Op1MoverNflip(bs.dimension, 1, True))

    validate_algorithm_on_leadingones(create)


def test_fea1plus1_on_leadingones_large_range() -> None:
    """Validate the (1+1)-FEA on the LeadingOnes problem with larger bounds."""

    def create(bs: BitStrings, objective: Objective) -> FEA1plus1:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        objective.lower_bound = __lb  # type: ignore
        objective.upper_bound = __ub  # type: ignore
        return FEA1plus1(Op0Random(), Op1MoverNflip(bs.dimension, 1, True))

    validate_algorithm_on_leadingones(create)


def test_h_log() -> None:
    """Test whether the history table is properly logged."""
    n = 10
    space = BitStrings(n)
    problem = OneMax(n)
    algorithm = FEA1plus1(Op0Random(), Op1Flip1())

    with temp_file() as tf:
        ex = Execution()
        ex.set_solution_space(space)
        ex.set_objective(problem)
        ex.set_algorithm(algorithm)
        ex.set_rand_seed(199)
        ex.set_log_file(tf)
        ex.set_max_fes(10)
        with ex.execute() as process:
            end_result = process.create()
            process.get_copy_of_best_y(end_result)

        lines = tf.read_all_str().splitlines()
        assert lines[-1] == "END_H"
        vals = [int(s) for s in lines[-2].split(";")]
        assert len(vals) % 2 == 0
        assert lines[-2] == "3;2;4;3;5;4;6;4;7;2;8;2;9;1"
        assert all(i >= 0 for i in vals)
        assert lines[-3] == "BEGIN_H"
