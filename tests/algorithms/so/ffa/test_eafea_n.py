"""Test the EAFEA-N."""
from pycommons.io.temp import temp_file

from moptipy.algorithms.so.ffa.eafea_n import EAFEAN
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
    """Get a mock lower bound."""
    return -1_000_000_000_000_000


def __ub() -> int:
    """Get a mock upper bound."""
    return 1_000_000_000_000_000


def test_eafean_on_jssp() -> None:
    """Validate the EAFEA-N on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> EAFEAN:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        return EAFEAN(Op0Shuffle(search_space), Op1Swap2(), False)

    validate_algorithm_on_jssp(create)


def test_eafean_on_onemax() -> None:
    """Validate the EAFEA-N on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> EAFEAN:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return EAFEAN(Op0Random(), Op1MoverNflip(bs.dimension, 1, True),
                      False)

    validate_algorithm_on_onemax(create)


def test_eafean_on_onemax_with_large_range() -> None:
    """Validate the EAFEA-N on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> EAFEAN:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        objective.lower_bound = __lb  # type: ignore
        objective.upper_bound = __ub  # type: ignore
        return EAFEAN(Op0Random(), Op1MoverNflip(bs.dimension, 1, True), True)

    validate_algorithm_on_onemax(create)


def test_eafean_on_leadingones() -> None:
    """Validate the EAFEA-N on the LeadingOnes problem."""

    def create(bs: BitStrings, objective: Objective) -> EAFEAN:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return EAFEAN(Op0Random(), Op1MoverNflip(bs.dimension, 1, True), True)

    validate_algorithm_on_leadingones(create)


def test_eafean_on_leadingones_large_range() -> None:
    """Validate the EAFEA-N on the LeadingOnes problem with larger bounds."""

    def create(bs: BitStrings, objective: Objective) -> EAFEAN:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        objective.lower_bound = __lb  # type: ignore
        objective.upper_bound = __ub  # type: ignore
        return EAFEAN(Op0Random(), Op1MoverNflip(bs.dimension, 1, True))

    validate_algorithm_on_leadingones(create)


def test_h_log() -> None:
    """Test whether the history table is properly logged."""
    n = 10
    space = BitStrings(n)
    problem = OneMax(n)
    algorithm = EAFEAN(Op0Random(), Op1Flip1(), True)

    with temp_file() as tf:
        ex = Execution()
        ex.set_solution_space(space)
        ex.set_objective(problem)
        ex.set_algorithm(algorithm)
        ex.set_rand_seed(1599)
        ex.set_log_file(tf)
        ex.set_max_fes(10)
        with ex.execute() as process:
            end_result = process.create()
            process.get_copy_of_best_y(end_result)

        lines = tf.read_all_str().splitlines()
        assert lines[-1] == "END_H"
        assert lines[-2] == "4;6;;9;;3"
        assert lines[-3] == "BEGIN_H"
