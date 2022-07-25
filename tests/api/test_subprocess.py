"""Test the Sub-Process API."""
from typing import Union, Final, Callable

from numpy.random import Generator

from moptipy.algorithms.ea import EA
from moptipy.algorithms.rls import RLS
from moptipy.api.algorithm import Algorithm2
from moptipy.api.execution import Execution
from moptipy.api.process import Process
from moptipy.api.subprocesses import from_starting_point, for_fes, \
    without_should_terminate
from moptipy.examples.bitstrings.ising1d import Ising1d
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.spaces.bitstrings import BitStrings


class MyAlgorithm(Algorithm2):
    """The dummy algorithm"""

    def __init__(self) -> None:
        super().__init__("dummy", Op0Random(), Op1Flip1(), Op2Uniform())
        self.ea = EA(self.op0, self.op1, self.op2, 10, 10, 0.3)
        self.rls = RLS(self.op0, self.op1, seeded=True)

    def solve(self, process: Process) -> None:
        """Apply an EA for 100 FEs, followed by RLS."""
        fnew: Union[int, float]
        fes: int
        x1 = process.create()
        x2 = process.create()

        assert not process.has_best()
        with for_fes(process, 100) as z:
            assert not z.has_best()
            assert z.get_consumed_fes() == 0
            self.ea.solve(z)
            fnew = z.get_best_f()
            assert fnew >= 0
            fes = z.get_consumed_fes()
            assert fes > 0
            assert (fnew == 0) or (fes == 100)
            assert (fnew >= 0) and (fes <= 100)
            assert z.has_best()
            z.get_copy_of_best_x(x1)
        assert process.has_best()
        process.get_copy_of_best_x(x2)
        assert process.is_equal(x1, x2)
        assert process.get_consumed_fes() == fes

        assert process.get_best_f() == fnew
        if fnew > 0:
            assert process.evaluate(x1) == fnew

        fnew2: Union[int, float]
        fes2: int
        with for_fes(from_starting_point(process, x1, fnew), 100) as z:
            assert z.has_best()
            assert z.get_best_f() == fnew
            assert z.get_consumed_fes() == 0
            self.rls.solve(z)
            fnew2 = z.get_best_f()
            fes2 = z.get_consumed_fes()
            assert fes2 > 0
            assert (fnew2 == 0) or (fes2 == 100)
            assert (fnew2 >= 0) and (fes2 <= 100)
            assert fnew2 <= fnew

        allfes = process.get_consumed_fes()
        assert allfes == 1 + fes + fes2
        assert process.get_best_f() == fnew2
        if fnew2 > 0:
            assert process.evaluate(x1) == fnew2


def test_from_start_for_fes():
    """Slice off some FEs from a process to apply another process."""
    v = BitStrings(32)
    f = Ising1d(32)

    exp = Execution()
    exp.set_algorithm(MyAlgorithm())
    exp.set_solution_space(v)
    exp.set_objective(f)
    with exp.execute() as p:
        assert p.has_best()
        assert p.get_best_f() >= 0
        assert (p.get_best_f() == 0) or (p.get_consumed_fes() == 202)
        assert (p.get_best_f() >= 0) and (p.get_consumed_fes() <= 202)


class MyAlgorithm2(Algorithm2):
    """The dummy algorithm"""

    def __init__(self) -> None:
        super().__init__("dummy", Op0Random(), Op1Flip1(), Op2Uniform())

    def solve(self, process: Process) -> None:
        without_should_terminate(self._solve, process)

    def _solve(self, process: Process) -> None:
        # Create records for old and new point in the search space.
        best_x = process.create()  # record for best-so-far solution
        new_x = process.create()  # record for new solution
        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to save time.
        evaluate: Final[Callable] = process.evaluate  # the objective
        op1: Final[Callable] = self.op1.op1  # the unary operator

        # Start at a random point in the search space and evaluate it.
        self.op0.op0(random, best_x)  # Create 1 solution randomly and
        best_f: Union[int, float] = evaluate(best_x)  # evaluate it.

        while True:  # Never quit!
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f: Union[int, float] = evaluate(new_x)
            if new_f <= best_f:  # new_x is not worse than best_x?
                best_f = new_f  # Store its objective value.
                best_x, new_x = new_x, best_x  # Swap best and new.


def test_without_should_terminate():
    """Test an algorithm that never terminates."""
    v = BitStrings(32)
    f = Ising1d(32)

    exp = Execution()
    exp.set_algorithm(MyAlgorithm2())
    exp.set_solution_space(v)
    exp.set_objective(f)
    exp.set_max_fes(100)
    with exp.execute() as p:
        assert p.has_best()
        assert p.get_best_f() >= 0
        assert (p.get_best_f() == 0) or (p.get_consumed_fes() == 100)
        assert (p.get_best_f() >= 0) and (p.get_consumed_fes() <= 100)
