"""Test the `_process_ss`."""

from os.path import exists, isfile

from numpy.random import Generator, default_rng

from moptipy.algorithms.so.ea import EA
from moptipy.api.algorithm import Algorithm, Algorithm0
from moptipy.api.encoding import Encoding
from moptipy.api.execution import Execution
from moptipy.api.objective import Objective
from moptipy.api.process import Process
from moptipy.api.space import Space
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swapn import Op1SwapN
from moptipy.operators.permutations.op2_gap import (
    Op2GeneralizedAlternatingPosition,
)
from moptipy.operators.permutations.op2_ox2 import Op2OrderBased
from moptipy.spaces.permutations import Permutations
from moptipy.utils.temp import TempFile
from moptipy.utils.types import type_name_of


def test_process_ss_log_log() -> None:
    """Test the `_process_ss_log_log` with logging."""
    random: Generator = default_rng()
    instance: Instance = Instance.from_resource("orb06")
    search_space: Permutations = Permutations.with_repetitions(
        instance.jobs, instance.machines)
    solution_space: Space = GanttSpace(instance)
    encoding: Encoding = OperationBasedEncoding(instance)
    objective: Objective = Makespan(instance)
    algorithm: Algorithm = EA(
        Op0Shuffle(search_space), Op1SwapN(),
        Op2OrderBased(search_space),
        int(random.integers(3, 16)),
        int(random.integers(3, 16)),
        float(random.uniform(0.1, 0.9)))

    with TempFile.create() as tf:
        assert exists(tf)
        assert isfile(tf)
        with Execution()\
                .set_search_space(search_space)\
                .set_solution_space(solution_space)\
                .set_encoding(encoding)\
                .set_objective(objective)\
                .set_algorithm(algorithm)\
                .set_goal_f(int(round(1.5 * objective.lower_bound())))\
                .set_max_time_millis(60000)\
                .set_log_file(tf)\
                .set_log_improvements(True)\
                .execute() as process:
            assert type_name_of(process) \
                   == "moptipy.api._process_ss_log._ProcessSSLog"
            assert str(process) == "LoggingProcessWithSearchSpace"
            assert process.has_best()
            assert objective.lower_bound() <= process.get_best_f() \
                   <= objective.upper_bound()
            assert process.get_consumed_fes() > 0
            assert process.get_max_time_millis() == 60000
            assert process.get_max_fes() is None
            x = search_space.create()
            process.get_copy_of_best_x(x)
            search_space.validate(x)
            y = solution_space.create()
            process.get_copy_of_best_y(y)
            solution_space.validate(y)

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
        assert data[i] == "BEGIN_RESULT_X"
        j = data.index("END_RESULT_X")
        assert j > i + 1
        i = j + 1
        assert data[i] == "BEGIN_RESULT_Y"
        j = data.index("END_RESULT_Y")
        assert j > i + 1
        assert j == len(data) - 1


def test_process_ss_log_log_all() -> None:
    """Test the `_process_ss_log_log` with logging."""
    random: Generator = default_rng()
    instance: Instance = Instance.from_resource("swv15")
    search_space: Permutations = Permutations.with_repetitions(
        instance.jobs, instance.machines)
    solution_space: Space = GanttSpace(instance)
    encoding: Encoding = OperationBasedEncoding(instance)
    objective: Objective = Makespan(instance)
    algorithm: Algorithm = EA(
        Op0Shuffle(search_space), Op1SwapN(),
        Op2GeneralizedAlternatingPosition(search_space),
        int(random.integers(3, 16)),
        int(random.integers(3, 16)),
        float(random.uniform(0.1, 0.9)))

    with TempFile.create() as tf:
        assert exists(tf)
        assert isfile(tf)
        with Execution()\
                .set_search_space(search_space)\
                .set_solution_space(solution_space)\
                .set_encoding(encoding)\
                .set_objective(objective)\
                .set_algorithm(algorithm)\
                .set_max_fes(100)\
                .set_log_file(tf)\
                .set_log_improvements(True)\
                .set_log_all_fes(True)\
                .execute() as process:
            assert type_name_of(process) \
                   == "moptipy.api._process_ss_log._ProcessSSLog"
            assert str(process) == "LoggingProcessWithSearchSpace"
            assert process.has_best()
            assert process.get_max_fes() == 100
            assert process.get_max_time_millis() is None
            assert objective.lower_bound() <= process.get_best_f() \
                   <= objective.upper_bound()
            assert 0 < process.get_consumed_fes() <= 100
            x = search_space.create()
            process.get_copy_of_best_x(x)
            search_space.validate(x)
            y = solution_space.create()
            process.get_copy_of_best_y(y)
            solution_space.validate(y)

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
        assert data[i] == "BEGIN_RESULT_X"
        j = data.index("END_RESULT_X")
        assert j > i + 1
        i = j + 1
        assert data[i] == "BEGIN_RESULT_Y"
        j = data.index("END_RESULT_Y")
        assert j > i + 1
        assert j == len(data) - 1


class _RA(Algorithm0):
    """The one-max algorithm."""

    def __init__(self, op0: Op0Shuffle, x, y: Gantt, f: Makespan,
                 e: OperationBasedEncoding):
        """Initialize."""
        super().__init__("om", op0)
        self.f: Makespan = f
        self.x = x
        self.y: Gantt = y
        self.e: OperationBasedEncoding = e

    def solve(self, process: Process) -> None:
        """Solve."""
        r = process.get_random()
        i: int = 0
        while not process.should_terminate():
            self.op0.op0(r, self.x)
            self.e.decode(self.x, self.y)
            f = self.f.evaluate(self.y)
            process.register(self.x, f)
            assert process.get_consumed_time_millis() > 0
            i = i + 1
            assert process.get_consumed_fes() == i


def test_process_ss_log_log_reg() -> None:
    """Test the `_process_ss_log_log` with logging."""
    instance: Instance = Instance.from_resource("ta33")
    search_space: Permutations = Permutations.with_repetitions(
        instance.jobs, instance.machines)
    solution_space: Space = GanttSpace(instance)
    encoding: OperationBasedEncoding = OperationBasedEncoding(instance)
    objective: Makespan = Makespan(instance)
    algorithm: Algorithm = _RA(Op0Shuffle(search_space),
                               search_space.create(), solution_space.create(),
                               objective, encoding)

    with TempFile.create() as tf:
        assert exists(tf)
        assert isfile(tf)
        with Execution()\
                .set_search_space(search_space)\
                .set_solution_space(solution_space)\
                .set_encoding(encoding)\
                .set_objective(objective)\
                .set_algorithm(algorithm)\
                .set_max_fes(100)\
                .set_log_file(tf)\
                .set_log_improvements(True)\
                .execute() as process:
            assert type_name_of(process) \
                   == "moptipy.api._process_ss_log._ProcessSSLog"
            assert str(process) == "LoggingProcessWithSearchSpace"
            assert process.has_best()
            assert objective.lower_bound() <= process.get_best_f() \
                   <= objective.upper_bound()
            assert process.get_consumed_fes() > 0
            x = search_space.create()
            process.get_copy_of_best_x(x)
            search_space.validate(x)
            y = solution_space.create()
            process.get_copy_of_best_y(y)
            solution_space.validate(y)

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
        assert data[i] == "BEGIN_RESULT_X"
        j = data.index("END_RESULT_X")
        assert j > i + 1
        i = j + 1
        assert data[i] == "BEGIN_RESULT_Y"
        j = data.index("END_RESULT_Y")
        assert j > i + 1
        assert j == len(data) - 1


def test_process_ss_log_log_all_reg() -> None:
    """Test the `_process_ss_log_log` with logging."""
    instance: Instance = Instance.from_resource("ta33")
    search_space: Permutations = Permutations.with_repetitions(
        instance.jobs, instance.machines)
    solution_space: Space = GanttSpace(instance)
    encoding: OperationBasedEncoding = OperationBasedEncoding(instance)
    objective: Makespan = Makespan(instance)
    algorithm: Algorithm = _RA(Op0Shuffle(search_space),
                               search_space.create(), solution_space.create(),
                               objective, encoding)

    with TempFile.create() as tf:
        assert exists(tf)
        assert isfile(tf)
        with Execution()\
                .set_search_space(search_space)\
                .set_solution_space(solution_space)\
                .set_encoding(encoding)\
                .set_objective(objective)\
                .set_algorithm(algorithm)\
                .set_max_fes(100)\
                .set_log_file(tf)\
                .set_log_all_fes(True)\
                .execute() as process:
            assert type_name_of(process) \
                   == "moptipy.api._process_ss_log._ProcessSSLog"
            assert str(process) == "LoggingProcessWithSearchSpace"
            assert process.has_best()
            assert objective.lower_bound() <= process.get_best_f() \
                   <= objective.upper_bound()
            assert process.get_consumed_fes() > 0
            x = search_space.create()
            process.get_copy_of_best_x(x)
            search_space.validate(x)
            y = solution_space.create()
            process.get_copy_of_best_y(y)
            solution_space.validate(y)

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
        assert data[i] == "BEGIN_RESULT_X"
        j = data.index("END_RESULT_X")
        assert j > i + 1
        i = j + 1
        assert data[i] == "BEGIN_RESULT_Y"
        j = data.index("END_RESULT_Y")
        assert j > i + 1
        assert j == len(data) - 1
