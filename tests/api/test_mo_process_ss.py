"""Test the `_mo_process_ss`."""

from os.path import exists, isfile

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.algorithms.mo.morls import MORLS
from moptipy.algorithms.mo.nsga2 import NSGA2
from moptipy.api.algorithm import Algorithm
from moptipy.api.encoding import Encoding
from moptipy.api.logging import FILE_SUFFIX
from moptipy.api.mo_archive import MOArchivePruner, MORecord
from moptipy.api.mo_execution import MOExecution
from moptipy.api.mo_problem import MOProblem
from moptipy.api.objective import Objective
from moptipy.api.space import Space
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.examples.jssp.worktime import Worktime
from moptipy.mo.archive.keep_farthest import KeepFarthest
from moptipy.mo.problem.weighted_sum import Prioritize, WeightedSum
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.operators.permutations.op2_ox2 import Op2OrderBased
from moptipy.spaces.permutations import Permutations
from moptipy.utils.temp import TempFile
from moptipy.utils.types import type_name_of


def test_mo_process_mo_ss_no_log() -> None:
    """Test the `_mo_process_ss` without logging."""
    random: Generator = default_rng()
    instance: Instance = Instance.from_resource("yn4")
    search_space: Permutations = Permutations.with_repetitions(
        instance.jobs, instance.machines)
    solution_space: Space = GanttSpace(instance)
    encoding: Encoding = OperationBasedEncoding(instance)
    f0: Objective = Worktime(instance)
    f1: Objective = Makespan(instance)
    problem: MOProblem = WeightedSum([f0, f1], [2, 3])
    pruner: MOArchivePruner = KeepFarthest(problem, [1, 0])
    algorithm: Algorithm = NSGA2(
        Op0Shuffle(search_space), Op1Swap2(), Op2OrderBased(search_space),
        int(random.integers(3, 22)), float(random.uniform(0.2, 0.8)))
    ams: int = int(random.integers(2, 8))
    with MOExecution()\
            .set_search_space(search_space)\
            .set_solution_space(solution_space)\
            .set_encoding(encoding)\
            .set_objective(problem)\
            .set_algorithm(algorithm)\
            .set_archive_pruner(pruner)\
            .set_max_fes(100)\
            .set_archive_max_size(ams)\
            .execute() as process:
        assert type_name_of(process) \
               == "moptipy.api._mo_process_ss._MOProcessSS"
        assert str(process) == "MOProcessWithSearchSpace"
        assert process.has_best()
        assert (2 * f0.lower_bound()) + (3 * f1.lower_bound()) \
               <= process.get_best_f() \
               <= (2 * f0.upper_bound()) + (3 * f1.upper_bound())
        assert 0 < process.get_consumed_fes() <= 100
        assert not process.has_log()
        archive: list[MORecord] = process.get_archive()
        for rec in archive:
            assert f0.lower_bound() <= rec.fs[0] <= f0.upper_bound()
            assert f1.lower_bound() <= rec.fs[1] <= f1.upper_bound()
        archive_len = len(archive)
        assert archive_len <= process.get_consumed_fes()
        x = search_space.create()
        process.get_copy_of_best_x(x)
        search_space.validate(x)
        y = solution_space.create()
        process.get_copy_of_best_y(y)
        solution_space.validate(y)
        fs = process.f_create()
        assert isinstance(fs, np.ndarray)
        assert len(fs) == problem.f_dimension()
        process.get_copy_of_best_fs(fs)
        fs2 = fs.copy()
        assert problem.f_evaluate(y, fs2) == process.get_best_f()
        assert np.array_equal(fs2, fs)


def test_mo_process_ss_log() -> None:
    """Test the `_mo_process_ss` without logging but log file."""
    instance: Instance = Instance.from_resource("ft10")
    search_space: Permutations = Permutations.with_repetitions(
        instance.jobs, instance.machines)
    solution_space: Space = GanttSpace(instance)
    encoding: Encoding = OperationBasedEncoding(instance)
    f0: Objective = Worktime(instance)
    f1: Objective = Makespan(instance)
    problem: MOProblem = Prioritize([f0, f1])
    algorithm: Algorithm = MORLS(
        Op0Shuffle(search_space), Op1Swap2())

    with TempFile.create(suffix=FILE_SUFFIX) as tf:
        assert exists(tf)
        assert isfile(tf)
        archive_len: int
        with MOExecution()\
                .set_search_space(search_space)\
                .set_solution_space(solution_space)\
                .set_encoding(encoding)\
                .set_objective(problem)\
                .set_algorithm(algorithm)\
                .set_max_fes(100)\
                .set_log_file(tf)\
                .execute() as process:
            assert type_name_of(process) \
                   == "moptipy.api._mo_process_ss._MOProcessSS"
            assert str(process) == "MOProcessWithSearchSpace"
            assert process.has_best()
            assert 0 < process.get_best_f() \
                   <= (f0.upper_bound() ** 2) + (f1.upper_bound() ** 2) \
                   + ((f1.upper_bound() + 1) * (1 + f0.upper_bound()))
            assert 0 < process.get_consumed_fes() <= 100
            assert process.has_log()
            archive: list[MORecord] = process.get_archive()
            for rec in archive:
                assert f0.lower_bound() <= rec.fs[0] <= f0.upper_bound()
                assert f1.lower_bound() <= rec.fs[1] <= f1.upper_bound()
            archive_len = len(archive)
            assert 0 < archive_len <= process.get_consumed_fes()
            x = search_space.create()
            process.get_copy_of_best_x(x)
            search_space.validate(x)
            y = solution_space.create()
            process.get_copy_of_best_y(y)
            solution_space.validate(y)
            fs = process.f_create()
            assert isinstance(fs, np.ndarray)
            assert len(fs) == problem.f_dimension()
            process.get_copy_of_best_fs(fs)
            fs2 = fs.copy()
            assert problem.f_evaluate(y, fs2) == process.get_best_f()
            assert np.array_equal(fs2, fs)

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

        for z in range(archive_len):
            i = j + 1
            assert data[i] == f"BEGIN_ARCHIVE_{z}_X"
            j = data.index(f"END_ARCHIVE_{z}_X")
            assert j > i + 1
            i = j + 1
            assert data[i] == f"BEGIN_ARCHIVE_{z}_Y"
            j = data.index(f"END_ARCHIVE_{z}_Y")
            assert j > i + 1

        i = j + 1
        assert data[i] == "BEGIN_ARCHIVE_QUALITIES"
        i = i + 1
        assert data[i] == "f;f0;f1"
        i += archive_len + 1
        assert data[i] == "END_ARCHIVE_QUALITIES"
        assert i == len(data) - 1
