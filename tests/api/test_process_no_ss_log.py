"""Test the `_process_no_ss_log`."""

from os.path import exists, isfile

from numpy.random import Generator, default_rng

from moptipy.algorithms.so.fea1plus1 import FEA1plus1
from moptipy.algorithms.so.hill_climber_with_restarts import (
    HillClimberWithRestarts,
)
from moptipy.api.algorithm import Algorithm, Algorithm0
from moptipy.api.execution import Execution
from moptipy.api.objective import Objective
from moptipy.api.process import Process
from moptipy.api.space import Space
from moptipy.examples.bitstrings.leadingones import LeadingOnes
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.examples.bitstrings.trap import Trap
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.temp import TempFile
from moptipy.utils.types import type_name_of


def test_process_no_ss_log_log() -> None:
    """Test the `_process_no_ss_log` with logging."""
    random: Generator = default_rng()
    dim: int = int(random.integers(5, 12))
    space: Space = BitStrings(dim)
    objective: Objective = Trap(dim)
    algorithm: Algorithm = FEA1plus1(Op0Random(), Op1Flip1())

    with TempFile.create() as tf:
        assert exists(tf)
        assert isfile(tf)

        with Execution()\
                .set_solution_space(space)\
                .set_objective(objective)\
                .set_algorithm(algorithm)\
                .set_max_time_millis(20)\
                .set_goal_f(dim - 2)\
                .set_max_fes(100)\
                .set_log_file(tf)\
                .set_log_improvements(True) \
                .execute() as process:
            assert type_name_of(process) \
                   == "moptipy.api._process_no_ss_log._ProcessNoSSLog"
            assert str(process) == "LoggingProcessWithoutSearchSpace"
            assert process.has_best()
            assert process.get_max_fes() == 100
            assert process.get_max_time_millis() == 20
            assert process.get_best_f() >= 0
            assert 0 <= process.get_best_f() <= dim
            assert 0 < process.get_consumed_time_millis() <= 1000
            assert 0 < process.get_consumed_fes() <= 100
            x = space.create()
            process.get_copy_of_best_x(x)
            space.validate(x)
            process.get_copy_of_best_y(x)
            space.validate(x)

        assert exists(tf)
        assert isfile(tf)
        data = tf.read_all_list()
        assert len(data) > 10
        assert data[0] == "BEGIN_PROGRESS"
        assert data[1] == "fes;timeMS;f"
        i = data.index("END_PROGRESS")
        assert i > 2
        i += 1
        assert data[i] == "BEGIN_STATE"
        i += 1
        j = data.index("END_STATE")
        assert j > i
        i = (j + 1)
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
        assert data[-1] == "END_RESULT_Y"


def test_process_no_ss_log_log_all() -> None:
    """Test the `_process_no_ss_log` with logging."""
    random: Generator = default_rng()
    dim: int = int(random.integers(3, 12))
    space: Space = BitStrings(dim)
    objective: Objective = LeadingOnes(dim)
    algorithm: Algorithm = HillClimberWithRestarts(
        Op0Random(), Op1Flip1(), int(round(dim * 1.6)))

    with TempFile.create() as tf:
        assert exists(tf)
        assert isfile(tf)

        with Execution()\
                .set_solution_space(space)\
                .set_objective(objective)\
                .set_algorithm(algorithm)\
                .set_goal_f(max(2, dim - 4))\
                .set_log_file(tf)\
                .set_log_improvements(True) \
                .set_log_all_fes(True) \
                .execute() as process:
            assert type_name_of(process) \
                   == "moptipy.api._process_no_ss_log._ProcessNoSSLog"
            assert str(process) == "LoggingProcessWithoutSearchSpace"
            assert process.has_best()
            assert process.get_best_f() >= 0
            assert 0 <= process.get_best_f() <= dim
            assert process.get_consumed_time_millis() > 0
            assert process.get_consumed_fes() > 0
            x = space.create()
            process.get_copy_of_best_x(x)
            space.validate(x)
            process.get_copy_of_best_y(x)
            space.validate(x)

        assert exists(tf)
        assert isfile(tf)
        data = tf.read_all_list()
        assert len(data) > 10
        assert data[0] == "BEGIN_PROGRESS"
        assert data[1] == "fes;timeMS;f"
        i = data.index("END_PROGRESS")
        assert i > 2
        i += 1
        assert data[i] == "BEGIN_STATE"
        i += 1
        j = data.index("END_STATE")
        assert j > i
        i = (j + 1)
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
        assert data[-1] == "END_RESULT_Y"


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

    with TempFile.create() as tf:
        assert exists(tf)
        assert isfile(tf)

        with Execution()\
                .set_solution_space(space)\
                .set_objective(objective)\
                .set_algorithm(algorithm)\
                .set_max_time_millis(20)\
                .set_goal_f(dim - 2)\
                .set_max_fes(100)\
                .set_log_file(tf)\
                .set_log_improvements(True) \
                .execute() as process:
            assert type_name_of(process) \
                   == "moptipy.api._process_no_ss_log._ProcessNoSSLog"
            assert str(process) == "LoggingProcessWithoutSearchSpace"
            assert process.has_best()
            assert process.get_max_fes() == 100
            assert process.get_max_time_millis() == 20
            assert process.get_best_f() >= 0
            assert 0 <= process.get_best_f() <= dim
            assert 0 < process.get_consumed_time_millis() <= 1000
            assert 0 < process.get_consumed_fes() <= 100
            x = space.create()
            process.get_copy_of_best_x(x)
            space.validate(x)
            process.get_copy_of_best_y(x)
            space.validate(x)

        assert exists(tf)
        assert isfile(tf)
        data = tf.read_all_list()
        assert len(data) > 10
        assert data[0] == "BEGIN_PROGRESS"
        assert data[1] == "fes;timeMS;f"
        i = data.index("END_PROGRESS")
        assert i > 2
        i += 1
        assert data[i] == "BEGIN_STATE"
        i += 1
        j = data.index("END_STATE")
        assert j > i
        i = (j + 1)
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
        assert data[-1] == "END_RESULT_Y"


def test_process_no_ss_no_log_all_register() -> None:
    """Test the `_process_no_ss` without logging."""
    random: Generator = default_rng()
    dim: int = int(random.integers(3, 12))
    space: Space = BitStrings(dim)
    objective: OneMax = OneMax(dim)
    algorithm: _OMA = _OMA(Op0Random(), objective)

    with TempFile.create() as tf:
        assert exists(tf)
        assert isfile(tf)

        with Execution()\
                .set_solution_space(space)\
                .set_objective(objective)\
                .set_algorithm(algorithm)\
                .set_max_time_millis(20)\
                .set_goal_f(dim - 2)\
                .set_max_fes(100)\
                .set_log_file(tf)\
                .set_log_all_fes(True) \
                .execute() as process:
            assert type_name_of(process) \
                   == "moptipy.api._process_no_ss_log._ProcessNoSSLog"
            assert str(process) == "LoggingProcessWithoutSearchSpace"
            assert process.has_best()
            assert process.get_max_fes() == 100
            assert process.get_max_time_millis() == 20
            assert process.get_best_f() >= 0
            assert 0 <= process.get_best_f() <= dim
            assert 0 < process.get_consumed_time_millis() <= 1000
            assert 0 < process.get_consumed_fes() <= 100
            x = space.create()
            process.get_copy_of_best_x(x)
            space.validate(x)
            process.get_copy_of_best_y(x)
            space.validate(x)

        assert exists(tf)
        assert isfile(tf)
        data = tf.read_all_list()
        assert len(data) > 10
        assert data[0] == "BEGIN_PROGRESS"
        assert data[1] == "fes;timeMS;f"
        i = data.index("END_PROGRESS")
        assert i > 2
        i += 1
        assert data[i] == "BEGIN_STATE"
        i += 1
        j = data.index("END_STATE")
        assert j > i
        i = (j + 1)
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
        assert data[-1] == "END_RESULT_Y"
