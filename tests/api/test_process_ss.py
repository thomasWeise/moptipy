"""Test the `_process_ss`."""

from os.path import exists, isfile

from moptipy.algorithms.random_walk import RandomWalk
from moptipy.algorithms.single_random_sample import SingleRandomSample
from moptipy.api.algorithm import Algorithm, Algorithm0
from moptipy.api.encoding import Encoding
from moptipy.api.execution import Execution
from moptipy.api.logging import FILE_SUFFIX
from moptipy.api.objective import Objective
from moptipy.api.process import Process
from moptipy.api.space import Space
from moptipy.examples.jssp.gantt import Gantt  # Gantt chart data structure
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.examples.jssp.worktime import Worktime
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.permutations import Permutations
from moptipy.utils.temp import TempFile
from moptipy.utils.types import type_name_of


def test_process_ss_no_log() -> None:
    """Test the `_process_ss` without logging."""
    instance: Instance = Instance.from_resource("dmu23")
    search_space: Permutations = Permutations.with_repetitions(
        instance.jobs, instance.machines)
    solution_space: Space = GanttSpace(instance)
    encoding: Encoding = OperationBasedEncoding(instance)
    objective: Objective = Worktime(instance)
    algorithm: Algorithm = SingleRandomSample(Op0Shuffle(search_space))

    with Execution()\
            .set_search_space(search_space)\
            .set_solution_space(solution_space)\
            .set_encoding(encoding)\
            .set_objective(objective)\
            .set_algorithm(algorithm)\
            .set_max_fes(100)\
            .execute() as process:
        assert type_name_of(process) \
               == "moptipy.api._process_ss._ProcessSS"
        assert str(process) == "ProcessWithSearchSpace"
        assert process.has_best()
        assert process.get_max_fes() == 100
        assert process.get_max_time_millis() is None
        assert objective.lower_bound() <= process.get_best_f() \
               <= objective.upper_bound()
        assert process.get_consumed_fes() == 1
        x = search_space.create()
        process.get_copy_of_best_x(x)
        search_space.validate(x)
        y = solution_space.create()
        process.get_copy_of_best_y(y)
        solution_space.validate(y)


def test_process_ss_log() -> None:
    """Test the `_process_ss` without logging."""
    instance: Instance = Instance.from_resource("la12")
    search_space: Permutations = Permutations.with_repetitions(
        instance.jobs, instance.machines)
    solution_space: Space = GanttSpace(instance)
    encoding: Encoding = OperationBasedEncoding(instance)
    objective: Objective = Makespan(instance)
    algorithm: Algorithm = RandomWalk(Op0Shuffle(search_space),
                                      Op1Swap2())
    with TempFile.create(suffix=FILE_SUFFIX) as tf:
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
                .execute() as process:
            assert type_name_of(process) \
                   == "moptipy.api._process_ss._ProcessSS"
            assert str(process) == "ProcessWithSearchSpace"
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
        assert data[i] == "BEGIN_RESULT_X"
        j = data.index("END_RESULT_X")
        assert j > i + 1
        i = j + 1
        assert data[i] == "BEGIN_RESULT_Y"
        j = data.index("END_RESULT_Y")
        assert j > i + 1
        assert j == len(data) - 1

        gantt = Gantt.from_log(tf)
        assert isinstance(gantt, Gantt)
        assert gantt.instance.name == instance.name


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
        while not process.should_terminate():
            self.op0.op0(r, self.x)
            self.e.decode(self.x, self.y)
            f = self.f.evaluate(self.y)
            process.register(self.x, f)


def test_process_ss_no_log_reg() -> None:
    """Test the `_process_ss` without logging but registering."""
    instance: Instance = Instance.from_resource("la30")
    search_space: Permutations = Permutations.with_repetitions(
        instance.jobs, instance.machines)
    solution_space: Space = GanttSpace(instance)
    encoding: OperationBasedEncoding = OperationBasedEncoding(instance)
    objective: Makespan = Makespan(instance)
    algorithm: Algorithm = _RA(Op0Shuffle(search_space),
                               search_space.create(), solution_space.create(),
                               objective, encoding)

    with Execution()\
            .set_search_space(search_space)\
            .set_solution_space(solution_space)\
            .set_encoding(encoding)\
            .set_objective(objective)\
            .set_algorithm(algorithm)\
            .set_max_fes(100)\
            .execute() as process:
        assert type_name_of(process) \
               == "moptipy.api._process_ss._ProcessSS"
        assert str(process) == "ProcessWithSearchSpace"
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
