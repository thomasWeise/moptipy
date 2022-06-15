"""Test the RLS."""
from typing import Callable, List, Final

from numpy.random import default_rng

from moptipy.algorithms.ea import EA
from moptipy.algorithms.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.spaces.bitstrings import BitStrings


def test_opoea_equals_rls():
    """Test whether the (1+1)-EA performs exactly as RLS."""

    problem: Final[OneMax] = OneMax(21)
    search_space: Final[BitStrings] = BitStrings(problem.n)
    op0: Final[Op0] = Op0Random()
    op1: Final[Op1] = Op1Flip1()
    op2: Final[Op2] = Op2Uniform()
    seed: Final[int] = int(default_rng().integers(1 << 62))

    opoea: Final[EA] = EA(op0, op1, op2, 1, 1, 0.0)
    rls: Final[RLS] = RLS(op0, op1)

    moves_opoea: Final[List[bool]] = []
    moves_rls: Final[List[bool]] = []

    evaluate: Final[Callable] = problem.evaluate

    def f_opoea(x) -> int:
        nonlocal moves_opoea
        nonlocal evaluate
        res = evaluate(x)
        moves_opoea.extend(x)
        return res

    def f_rls(x) -> int:
        nonlocal moves_rls
        nonlocal evaluate
        res = evaluate(x)
        moves_rls.extend(x)
        return res

    ex_opoea = Execution()
    ex_opoea.set_algorithm(opoea)
    ex_opoea.set_solution_space(search_space)
    problem.evaluate = f_opoea
    ex_opoea.set_objective(problem)
    ex_opoea.set_rand_seed(seed)
    with ex_opoea.execute() as _:
        pass

    ex_rls = Execution()
    ex_rls.set_algorithm(rls)
    ex_rls.set_solution_space(search_space)
    problem.evaluate = f_rls
    ex_rls.set_objective(problem)
    ex_rls.set_rand_seed(seed)
    with ex_rls.execute() as _:
        pass

    assert len(moves_rls) == len(moves_opoea)
    assert moves_opoea == moves_rls
