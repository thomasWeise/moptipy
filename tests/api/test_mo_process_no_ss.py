"""Test the `_mo_process_no_ss`."""

from os.path import exists, isfile

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.algorithms.mo.morls import MORLS
from moptipy.algorithms.mo.nsga2 import NSGA2
from moptipy.api.algorithm import Algorithm
from moptipy.api.mo_archive import MOArchivePruner, MORecord
from moptipy.api.mo_execution import MOExecution
from moptipy.api.mo_problem import MOProblem
from moptipy.api.mo_process import MOProcess
from moptipy.api.objective import Objective
from moptipy.api.space import Space
from moptipy.examples.bitstrings.leadingones import LeadingOnes
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.examples.bitstrings.zeromax import ZeroMax
from moptipy.mo.archive.keep_farthest import KeepFarthest
from moptipy.mo.problem.weighted_sum import Prioritize, WeightedSum
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.temp import TempFile
from moptipy.utils.types import type_name_of


def test_mo_process_no_ss_no_log() -> None:
    """Test the `_mo_process_no_ss` without logging."""
    random: Generator = default_rng()
    dim: int = int(random.integers(12, 40))

    space: Space = BitStrings(dim)
    f0: Objective = ZeroMax(dim)
    f1: Objective = OneMax(dim)
    problem: MOProblem = Prioritize([f0, f1])
    pruner: MOArchivePruner = KeepFarthest(problem, [1])
    algorithm: Algorithm = MORLS(
        Op0Random(),
        Op1MoverNflip(dim, int(random.integers(1, dim - 1))))
    ams = int(random.integers(2, 5))

    with MOExecution()\
            .set_solution_space(space)\
            .set_objective(problem)\
            .set_archive_pruner(pruner)\
            .set_archive_max_size(ams)\
            .set_archive_pruning_limit(int(ams + random.integers(0, 3)))\
            .set_algorithm(algorithm)\
            .set_max_fes(100)\
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
        fs = process.f_create()
        assert isinstance(fs, np.ndarray)
        assert len(fs) == problem.f_dimension()
        process.get_copy_of_best_fs(fs)
        fs2 = fs.copy()
        assert problem.f_evaluate(x, fs2) == process.get_best_f()
        assert np.array_equal(fs2, fs)


def test_mo_process_no_ss_log() -> None:
    """Test the `_mo_process_no_ss` with logging."""
    random: Generator = default_rng()
    dim: int = int(random.integers(13, 34))

    space: Space = BitStrings(dim)
    f0: Objective = LeadingOnes(dim)
    f1: Objective = ZeroMax(dim)
    problem: MOProblem = WeightedSum([f0, f1], [0.5, 0.5])
    pruner: MOArchivePruner = MOArchivePruner()
    algorithm: Algorithm = NSGA2(
        Op0Random(),
        Op1MoverNflip(dim, int(random.integers(1, dim - 1))),
        Op2Uniform(), int(random.integers(4, 16)),
        float(random.uniform(0.3, 0.7)))
    ams = int(random.integers(2, 5))

    with TempFile.create() as tf:
        assert exists(tf)
        assert isfile(tf)

        archive_len: int
        with MOExecution() \
                .set_solution_space(space) \
                .set_objective(problem) \
                .set_archive_pruner(pruner) \
                .set_archive_max_size(ams) \
                .set_archive_pruning_limit(int(ams + random.integers(0, 3))) \
                .set_algorithm(algorithm) \
                .set_max_time_millis(100)\
                .set_log_file(tf)\
                .execute() as process:
            assert isinstance(process, MOProcess)
            assert type_name_of(process) \
                   == "moptipy.api._mo_process_no_ss._MOProcessNoSS"
            assert str(process) == "MOProcessWithoutSearchSpace"
            assert process.has_best()
            assert process.get_max_time_millis() == 100
            assert process.get_max_fes() is None
            assert 0 <= process.get_best_f() <= dim + 0.000001
            assert process.get_consumed_fes() > 0
            assert process.get_consumed_time_millis() > 0
            archive: list[MORecord] = process.get_archive()
            for rec in archive:
                assert f0.lower_bound() <= rec.fs[0] <= f0.upper_bound()
                assert f1.lower_bound() <= rec.fs[1] <= f1.upper_bound()
            archive_len = len(archive)
            assert archive_len > 0
            x = space.create()
            process.get_copy_of_best_x(x)
            space.validate(x)
            process.get_copy_of_best_y(x)
            space.validate(x)
            fs = process.f_create()
            assert isinstance(fs, np.ndarray)
            assert len(fs) == problem.f_dimension()
            process.get_copy_of_best_fs(fs)
            fs2 = fs.copy()
            assert problem.f_evaluate(x, fs2) == process.get_best_f()
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
        assert data[i] == "BEGIN_RESULT_Y"
        j = data.index("END_RESULT_Y")
        assert j > i + 1

        for z in range(archive_len):
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
