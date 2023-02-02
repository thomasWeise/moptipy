"""Test the Sub-Process API."""
from collections.abc import Callable
from typing import Final

from numpy.random import Generator, default_rng

from moptipy.algorithms.so.ea import EA
from moptipy.algorithms.so.rls import RLS
from moptipy.api.algorithm import Algorithm, Algorithm0, Algorithm2
from moptipy.api.execution import Execution
from moptipy.api.mo_algorithm import MOAlgorithm
from moptipy.api.mo_archive import MOArchivePruner, MORecord
from moptipy.api.mo_execution import MOExecution
from moptipy.api.mo_problem import MOProblem
from moptipy.api.mo_process import MOProcess
from moptipy.api.objective import Objective
from moptipy.api.process import Process
from moptipy.api.space import Space
from moptipy.api.subprocesses import (
    for_fes,
    from_starting_point,
    get_remaining_fes,
    without_should_terminate,
)
from moptipy.examples.bitstrings.ising1d import Ising1d
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.examples.bitstrings.trap import Trap
from moptipy.mo.archive.keep_farthest import KeepFarthest
from moptipy.mo.problem.weighted_sum import Prioritize
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.operators.op0_forward import Op0Forward
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.types import type_name_of


class MyAlgorithm1(Algorithm2):
    """The dummy algorithm."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__("dummy", Op0Random(), Op1Flip1(), Op2Uniform())
        self.ea = EA(self.op0, self.op1, self.op2, 10, 10, 0.3)
        self.fwd = Op0Forward()
        self.rls = RLS(self.fwd, self.op1)

    def initialize(self) -> None:
        """Initialize."""
        super().initialize()
        self.ea.initialize()
        self.fwd.initialize()
        self.rls.initialize()

    def solve(self, process: Process) -> None:
        """Apply an EA for 100 FEs, followed by RLS."""
        fnew: int | float
        fes: int
        x1 = process.create()
        x2 = process.create()

        assert not process.has_best()
        assert get_remaining_fes(process) == 9_223_372_036_854_775_807
        with for_fes(process, 100) as z:
            assert str(z) == f"forFEs_100_{process}"
            assert not z.has_best()
            assert z.get_consumed_fes() == 0
            assert get_remaining_fes(z) == 100
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

        fnew2: int | float
        fes2: int
        with from_starting_point(process, x1, fnew) as z1:
            assert str(z1) == f"fromStart_{process}"
            with for_fes(z1, 100) as z:
                assert get_remaining_fes(z) == 100
                self.fwd.forward_to(z.get_copy_of_best_x)
                assert str(z) == \
                       f"forFEs_100_fromStart_{process}"
                assert z.has_best()
                assert z.get_best_f() == fnew
                assert z.get_consumed_fes() == 0
                self.rls.solve(z)
                self.fwd.stop_forwarding()
                fnew2 = z.get_best_f()
                fes2 = z.get_consumed_fes()
                assert fes2 > 0
                assert (fnew2 == 0) or (fes2 == 100)
                assert (fnew2 >= 0) and (fes2 <= 100)
                assert fnew2 <= fnew
                assert fes2 + get_remaining_fes(z) == 100

        allfes = process.get_consumed_fes()
        assert allfes == fes + fes2
        assert process.get_best_f() == fnew2
        if fnew2 > 0:
            assert process.evaluate(x1) == fnew2


def test_from_start_for_fes_with_drift() -> None:
    """Slice off some FEs from a process to apply another process."""
    v = BitStrings(32)
    f = Ising1d(32)

    exp = Execution()
    exp.set_algorithm(MyAlgorithm1())
    exp.set_solution_space(v)
    exp.set_objective(f)
    with exp.execute() as p:
        assert p.has_best()
        assert p.get_best_f() >= 0
        assert (p.get_best_f() == 0) or (p.get_consumed_fes() == 200)
        assert (p.get_best_f() >= 0) and (p.get_consumed_fes() <= 200)


class MyAlgorithm2(Algorithm2):
    """The dummy algorithm."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__("dummy", Op0Random(), Op1Flip1(), Op2Uniform())

    def solve(self, process: Process) -> None:
        """Solve the optimization problem."""
        without_should_terminate(self._solve, process)

    def _solve(self, process: Process) -> None:
        assert str(process).startswith("protect_")
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
        best_f: int | float = evaluate(best_x)  # evaluate it.

        while True:  # Never quit!
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f: int | float = evaluate(new_x)
            if new_f <= best_f:  # new_x is not worse than best_x?
                best_f = new_f  # Store its objective value.
                best_x, new_x = new_x, best_x  # Swap best and new.


def test_without_should_terminate() -> None:
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
        assert (p.get_best_f() == 0) or (get_remaining_fes(p) == 0)
        assert (p.get_best_f() >= 0) and (get_remaining_fes(p) >= 0)


class _OneMaxRegAlgo(Algorithm0):
    """The one-max algorithm."""

    def __init__(self, op0: Op0Random, f: OneMax):
        """Initialize."""
        super().__init__("om", op0)
        self.f: OneMax = f

    def solve(self, process: Process) -> None:
        """Solve."""
        x = process.create()
        r = process.get_random()
        while not process.should_terminate():
            self.op0.op0(r, x)
            f = self.f.evaluate(x)
            process.register(x, f)


class __RegisterForFEs(Algorithm):
    """The one-max algorithm."""

    def __init__(self, a: Algorithm, b: Algorithm):
        """Initialize."""
        self.a = a
        self.b = b
        self.repeat = False

    def initialize(self) -> None:
        """Initialize."""
        super().initialize()
        self.a.initialize()
        self.b.initialize()

    def solve(self, process: Process) -> None:
        """Solve."""
        assert process.get_consumed_fes() == 0
        mf = process.get_max_fes()
        assert mf == 100
        cf: int
        with for_fes(process, 50) as sp1:
            assert str(sp1) == f"forFEs_50_{process}"
            assert process.get_consumed_fes() == 0
            assert sp1.get_consumed_fes() == 0
            assert sp1.get_max_fes() == 50
            self.a.solve(sp1)
            cf = process.get_consumed_fes()
            assert 0 < cf <= 50
            assert cf == sp1.get_consumed_fes()
            assert cf + get_remaining_fes(sp1) == 50
        self.repeat = process.should_terminate()
        if self.repeat:
            return
        assert cf >= 50
        nf = mf - cf
        with for_fes(process, nf) as sp2:
            assert str(sp2) == f"forFEs_{nf}_{process}"
            assert process.get_consumed_fes() == cf
            assert sp2.get_consumed_fes() == 0
            assert sp2.get_max_fes() == nf
            self.b.solve(sp2)
            assert 0 < sp2.get_consumed_fes() <= nf
            assert cf < process.get_consumed_fes() <= mf


def test_for_fes_process_no_ss_no_log_reg_norm() -> None:
    """Test the `_process_no_ss` without logging."""
    random: Generator = default_rng()

    while True:
        dim: int = int(random.integers(6, 43))
        space: Space = BitStrings(dim)
        objective: OneMax = OneMax(dim)
        algorithm1: Algorithm = RLS(
            Op0Random(),
            Op1MoverNflip(dim, int(random.integers(1, dim - 1))))
        algorithm2: Algorithm = _OneMaxRegAlgo(Op0Random(), objective)
        algorithm: __RegisterForFEs = __RegisterForFEs(algorithm1, algorithm2)

        with Execution() \
                .set_solution_space(space) \
                .set_objective(objective) \
                .set_algorithm(algorithm) \
                .set_max_fes(100) \
                .execute() as process:
            assert type_name_of(process) \
                   == "moptipy.api._process_no_ss._ProcessNoSS"
            assert str(process) == "ProcessWithoutSearchSpace"
            assert process.has_best()
            assert process.get_max_fes() == 100
            assert process.get_max_time_millis() is None
            assert 0 <= process.get_best_f() <= dim
            assert 0 < process.get_consumed_fes() <= 100
            x = space.create()
            process.get_copy_of_best_x(x)
            space.validate(x)
            process.get_copy_of_best_y(x)
            space.validate(x)
        if not algorithm.repeat:
            return


class _MOAlgoForFEs(MOAlgorithm, Algorithm0):
    """The algorithm for multi-objective optimization."""

    def __init__(self, op0: Op0Random):
        """Initialize."""
        Algorithm0.__init__(self, "om", op0)

    def solve_mo(self, process: MOProcess) -> None:
        """Solve."""
        me = process.get_max_fes()
        with for_fes(process, me) as pp:
            assert str(pp) == f"forFEsMO_{me}_{process}"
            assert process.get_consumed_fes() == 0
            assert pp.get_consumed_fes() == 0
            assert pp.get_max_fes() == me
            self.__solve_mo(pp)
            assert 0 < pp.get_consumed_fes() <= me
            assert pp.get_consumed_fes() == process.get_consumed_fes()

    def __solve_mo(self, process: MOProcess) -> None:
        """Solve."""
        x = process.create()
        fs = process.f_create()
        r = process.get_random()
        while not process.should_terminate():
            self.op0.op0(r, x)
            process.evaluate(x)
            if process.should_terminate():
                return
            process.f_evaluate(x, fs)


def test_for_fes_mo_process_no_ss_no_log() -> None:
    """Test the `_mo_process_no_ss` without logging."""
    random: Generator = default_rng()
    dim: int = int(random.integers(12, 40))

    space: Space = BitStrings(dim)
    f0: Objective = Trap(dim)
    f1: Objective = OneMax(dim)
    problem: MOProblem = Prioritize([f0, f1])
    pruner: MOArchivePruner = KeepFarthest(problem, [1])
    algorithm: Algorithm = _MOAlgoForFEs(Op0Random())
    ams = int(random.integers(2, 5))

    with MOExecution() \
            .set_solution_space(space) \
            .set_objective(problem) \
            .set_archive_pruner(pruner) \
            .set_archive_max_size(ams) \
            .set_archive_pruning_limit(int(ams + random.integers(0, 3))) \
            .set_algorithm(algorithm) \
            .set_max_fes(100) \
            .execute() as process:
        assert isinstance(process, MOProcess)
        assert type_name_of(process) \
               == "moptipy.api._mo_process_no_ss._MOProcessNoSS"
        assert str(process) == "MOProcessWithoutSearchSpace"
        assert process.has_best()
        assert process.get_max_fes() == 100
        assert process.get_max_time_millis() is None
        assert 0 <= process.get_best_f() <= dim * dim * dim
        assert 0 < process.get_consumed_fes() <= 100
        archive: list[MORecord] = process.get_archive()
        for rec in archive:
            assert f0.lower_bound() <= rec.fs[0] <= f0.upper_bound()
            assert f1.lower_bound() <= rec.fs[1] <= f1.upper_bound()
        archive_len = len(archive)
        assert 0 < archive_len <= process.get_consumed_fes()
        x = space.create()
        process.get_copy_of_best_x(x)
        space.validate(x)
        process.get_copy_of_best_y(x)
        space.validate(x)


class _MOWithoutShouldTerminate(MOAlgorithm, Algorithm0):
    """The algorithm for multi-objective optimization."""

    def __init__(self, op0: Op0Random):
        """Initialize."""
        Algorithm0.__init__(self, "om", op0)

    def solve_mo(self, process: MOProcess) -> None:
        """Solve."""
        assert process.get_consumed_fes() == 0
        without_should_terminate(self.__solve_mo, process)
        assert 0 < process.get_consumed_fes() <= 100

    def __solve_mo(self, process: MOProcess) -> None:
        """Solve."""
        assert str(process).startswith("protectMO_")
        x = process.create()
        fs = process.f_create()
        r = process.get_random()
        while True:
            self.op0.op0(r, x)
            process.evaluate(x)
            if process.should_terminate():
                return
            process.f_evaluate(x, fs)


def test_without_should_terminate_mo_process_no_ss_no_log() -> None:
    """Test the `_mo_process_no_ss` without logging."""
    random: Generator = default_rng()
    dim: int = int(random.integers(12, 40))

    space: Space = BitStrings(dim)
    f0: Objective = Trap(dim)
    f1: Objective = OneMax(dim)
    problem: MOProblem = Prioritize([f0, f1])
    algorithm: Algorithm = _MOWithoutShouldTerminate(Op0Random())
    ams = int(random.integers(2, 5))

    with MOExecution() \
            .set_solution_space(space) \
            .set_objective(problem) \
            .set_archive_pruning_limit(int(ams + random.integers(0, 3))) \
            .set_algorithm(algorithm) \
            .set_max_fes(100) \
            .execute() as process:
        assert isinstance(process, MOProcess)
        assert type_name_of(process) \
               == "moptipy.api._mo_process_no_ss._MOProcessNoSS"
        assert str(process) == "MOProcessWithoutSearchSpace"
        assert process.has_best()
        assert process.get_max_fes() == 100
        assert process.get_max_time_millis() is None
        assert 0 <= process.get_best_f() <= dim * dim * dim
        assert 0 < process.get_consumed_fes() <= 100
        assert get_remaining_fes(process) + process.get_consumed_fes() == 100
        archive: list[MORecord] = process.get_archive()
        for rec in archive:
            assert f0.lower_bound() <= rec.fs[0] <= f0.upper_bound()
            assert f1.lower_bound() <= rec.fs[1] <= f1.upper_bound()
        archive_len = len(archive)
        assert 0 < archive_len <= process.get_consumed_fes()
        x = space.create()
        process.get_copy_of_best_x(x)
        space.validate(x)
        process.get_copy_of_best_y(x)
        space.validate(x)
