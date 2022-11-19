"""Test the `_process_no_ss`."""

from os.path import exists, isfile

from numpy.random import Generator, default_rng

from moptipy.algorithms.so.ea import EA
from moptipy.algorithms.so.rls import RLS
from moptipy.api.algorithm import Algorithm, Algorithm0
from moptipy.api.execution import Execution
from moptipy.api.objective import Objective
from moptipy.api.process import Process
from moptipy.api.space import Space
from moptipy.examples.bitstrings.ising1d import Ising1d
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.temp import TempFile
from moptipy.utils.types import type_name_of


def test_process_no_ss_no_log() -> None:
    """Test the `_process_no_ss` without logging."""
    random: Generator = default_rng()
    dim: int = int(random.integers(3, 12))
    space: Space = BitStrings(dim)
    objective: Objective = OneMax(dim)
    algorithm: Algorithm = RLS(
        Op0Random(),
        Op1MoverNflip(dim, int(random.integers(1, dim - 1))))

    with Execution()\
            .set_solution_space(space)\
            .set_objective(objective)\
            .set_algorithm(algorithm)\
            .set_max_fes(100)\
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


def test_process_no_ss_log() -> None:
    """Test the `_process_no_ss` with logging."""
    random: Generator = default_rng()
    dim: int = int(random.integers(3, 12))
    space: Space = BitStrings(dim)
    objective: Objective = Ising1d(dim)
    algorithm: Algorithm = EA(Op0Random(), Op1Flip1(), Op2Uniform())

    with TempFile.create() as tf:
        assert exists(tf)
        assert isfile(tf)

        with Execution()\
                .set_solution_space(space)\
                .set_objective(objective)\
                .set_algorithm(algorithm)\
                .set_max_time_millis(20)\
                .set_log_file(tf)\
                .execute() as process:
            assert type_name_of(process) \
                   == "moptipy.api._process_no_ss._ProcessNoSS"
            assert str(process) == "ProcessWithoutSearchSpace"
            assert process.has_best()
            assert process.get_best_f() >= 0
            assert process.get_max_time_millis() == 20
            assert process.get_max_fes() is None
            assert 0 <= process.get_best_f() <= dim
            assert 0 < process.get_consumed_time_millis() <= 1000
            x = space.create()
            process.get_copy_of_best_x(x)
            space.validate(x)
            process.get_copy_of_best_y(x)
            space.validate(x)

        assert exists(tf)
        assert isfile(tf)
        data = tf.read_all_list()
        assert len(data) > 10
        assert data[0] == "BEGIN_STATE"
        i = data.index("END_STATE")
        assert i > 0
        i += 1
        assert data[i] == "BEGIN_SETUP"
        j = data.index("END_SETUP")
        assert j > i + 1
        i = j + 1
        assert data[i] == "BEGIN_SYS_INFO"
        j = data.index("END_SYS_INFO")
        assert j > i + 1
        i = j + 1
        assert data[i] == "BEGIN_RESULT_Y"
        j = data.index("END_RESULT_Y")
        assert j > i + 1
        assert j == len(data) - 1


class ImmediateErrorAlgo(Algorithm):
    """The error algorithm."""

    def solve(self, process: Process) -> None:
        """Solve the problem, but throw an error immediately."""
        raise ValueError("Hah!")


def test_process_no_ss_log_with_immediate_error() -> None:
    """Test the `_process_no_ss` with logging."""
    random: Generator = default_rng()
    dim: int = int(random.integers(3, 12))
    space: Space = BitStrings(dim)
    objective: Objective = Ising1d(dim)
    algorithm: Algorithm = ImmediateErrorAlgo()

    with TempFile.create() as tf:
        assert exists(tf)
        assert isfile(tf)

        got_error: bool = False
        try:
            with Execution()\
                    .set_solution_space(space)\
                    .set_objective(objective)\
                    .set_algorithm(algorithm)\
                    .set_max_fes(12)\
                    .set_log_file(tf)\
                    .execute() as process:
                assert type_name_of(process) \
                       == "moptipy.api._process_no_ss._ProcessNoSS"
                assert str(process) == "ProcessWithoutSearchSpace"
                assert not process.has_best()
                assert process.get_max_fes() == 12
                assert process.get_consumed_fes() == 0
        except ValueError:
            got_error = True

        assert got_error
        assert exists(tf)
        assert isfile(tf)
        data = tf.read_all_list()
        assert len(data) > 10
        assert data[0] == "BEGIN_STATE"
        i = data.index("END_STATE")
        assert i > 0
        i += 1
        assert data[i] == "BEGIN_SETUP"
        j = data.index("END_SETUP")
        assert j > i + 1
        i = j + 1
        assert data[i] == "BEGIN_SYS_INFO"
        j = data.index("END_SYS_INFO")
        assert j > i + 1
        i = j + 1
        assert data[i] == "BEGIN_ERROR_IN_RUN"
        j = data.index("END_ERROR_IN_RUN")
        assert j > i + 1
        i = j + 1
        assert data[i] == "BEGIN_ERROR_TIMING"
        j = data.index("END_ERROR_TIMING")
        assert j > i + 1
        assert j == len(data) - 1


class ErrorAlgoDelayed(Algorithm0):
    """The error algorithm."""

    def solve(self, process: Process) -> None:
        """Solve the problem, but throw an error immediately."""
        x = process.create()
        self.op0.op0(process.get_random(), x)
        process.evaluate(x)
        raise ValueError("Hah!")


def test_process_no_ss_log_with_error_after_evaluation() -> None:
    """Test the `_process_no_ss` with logging and a later error."""
    random: Generator = default_rng()
    dim: int = int(random.integers(3, 12))
    space: Space = BitStrings(dim)
    objective: Objective = Ising1d(dim)
    algorithm: Algorithm = ErrorAlgoDelayed("error_delayed", Op0Random())

    with TempFile.create() as tf:
        assert exists(tf)
        assert isfile(tf)

        got_error: bool = False
        try:
            with Execution() \
                    .set_solution_space(space) \
                    .set_objective(objective) \
                    .set_algorithm(algorithm) \
                    .set_max_fes(20) \
                    .set_log_file(tf) \
                    .execute() as process:
                assert type_name_of(process) \
                       == "moptipy.api._process_no_ss._ProcessNoSS"
                assert str(process) == "ProcessWithoutSearchSpace"
                assert process.has_best()
                assert process.get_best_f() >= 0
                assert 0 <= process.get_best_f() <= dim
                assert process.get_consumed_fes() == 1
                x = space.create()
                process.get_copy_of_best_x(x)
                space.validate(x)
                process.get_copy_of_best_y(x)
                space.validate(x)
        except ValueError:
            got_error = True

        assert got_error
        assert exists(tf)
        assert isfile(tf)
        data = tf.read_all_list()
        assert len(data) > 10
        assert data[0] == "BEGIN_STATE"
        i = data.index("END_STATE")
        assert i > 0
        i += 1
        assert data[i] == "BEGIN_SETUP"
        j = data.index("END_SETUP")
        assert j > i + 1
        i = j + 1
        assert data[i] == "BEGIN_SYS_INFO"
        j = data.index("END_SYS_INFO")
        assert j > i + 1
        i = j + 1
        assert data[i] == "BEGIN_RESULT_Y"
        j = data.index("END_RESULT_Y")
        assert j > i + 1
        i = j + 1
        assert data[i] == "BEGIN_ERROR_IN_RUN"
        j = data.index("END_ERROR_IN_RUN")
        assert j > i + 1
        assert j == (len(data) - 1)


class _OMA(Algorithm0):
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


def test_process_no_ss_no_log_register() -> None:
    """Test the `_process_no_ss` without logging."""
    random: Generator = default_rng()
    dim: int = int(random.integers(3, 12))
    space: Space = BitStrings(dim)
    objective: OneMax = OneMax(dim)
    algorithm: _OMA = _OMA(Op0Random(), objective)

    with Execution()\
            .set_solution_space(space)\
            .set_objective(objective)\
            .set_algorithm(algorithm)\
            .set_max_fes(100)\
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
