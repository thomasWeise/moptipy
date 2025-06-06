"""Test the execution of an experiment and parsing the log files the JSSP."""
import numpy as np
from pycommons.io.temp import temp_dir, temp_file
from pycommons.math.sample_statistics import SampleStatistics
from pycommons.processes.caller import is_ci_run

import moptipy.evaluation.base as bs
from moptipy.algorithms.single_random_sample import SingleRandomSample
from moptipy.algorithms.so.hill_climber import HillClimber
from moptipy.api import logging
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_results import from_csv as er_from_csv
from moptipy.evaluation.end_results import from_logs as er_from_logs
from moptipy.evaluation.end_results import to_csv as er_to_csv
from moptipy.evaluation.end_statistics import EndStatistics
from moptipy.evaluation.end_statistics import create as es_create
from moptipy.evaluation.end_statistics import from_csv as es_from_csv
from moptipy.evaluation.end_statistics import (
    from_end_results as es_from_end_results,
)
from moptipy.evaluation.end_statistics import to_csv as es_to_csv
from moptipy.evaluation.ert import compute_single_ert
from moptipy.evaluation.ert import create as ert_create
from moptipy.evaluation.progress import Progress
from moptipy.evaluation.progress import from_logs as pr_from_logs
from moptipy.evaluation.stat_run import StatRun
from moptipy.evaluation.stat_run import create as sr_create
from moptipy.evaluation.stat_run import from_progress as st_from_progress
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.permutations import Permutations

instances = [lambda: Instance.from_resource("dmu21"),
             lambda: Instance.from_resource("abz8"),
             lambda: Instance.from_resource("la24")]


def algo_1(inst) -> Execution:
    """The first algorithm."""
    ss = Permutations.with_repetitions(inst.jobs, inst.machines)
    sos = GanttSpace(inst)
    op0 = Op0Shuffle(ss)
    op1 = Op1Swap2()
    algo = HillClimber(op0, op1)
    ex = Execution()
    ex.set_algorithm(algo)
    ex.set_search_space(ss)
    ex.set_solution_space(sos)
    ex.set_encoding(OperationBasedEncoding(inst))
    ex.set_objective(Makespan(inst))
    ex.set_max_fes(300)
    ex.set_log_improvements()
    return ex


def algo_2(inst) -> Execution:
    """The second algorithm."""
    ss = Permutations.with_repetitions(inst.jobs, inst.machines)
    sos = GanttSpace(inst)
    op0 = Op0Shuffle(ss)
    algo = SingleRandomSample(op0)
    ex = Execution()
    ex.set_algorithm(algo)
    ex.set_search_space(ss)
    ex.set_solution_space(sos)
    ex.set_encoding(OperationBasedEncoding(inst))
    ex.set_objective(Makespan(inst))
    ex.set_max_fes(300)
    ex.set_log_improvements()
    return ex


def test_experiment_jssp() -> None:
    """Run a test experiment on the JSSP."""
    with temp_dir() as base_dir:

        run_experiment(instances=instances,
                       setups=[algo_1, algo_2],
                       n_runs=4,
                       base_dir=base_dir)

        results: list[EndResult] = list(er_from_logs(base_dir))
        assert len(results) == (4 * 2 * 3)
        results.sort()

        for err in results:
            assert err.path_to_file(base_dir).is_file()

        for i in range(12):
            assert results[i].algorithm == "1rs"
        for i in range(12, 24):
            assert results[i].algorithm == "hc_swap2"

        for i in range(4):
            assert results[i].instance == "abz8"
        for i in range(4, 8):
            assert results[i].instance == "dmu21"
        for i in range(8, 12):
            assert results[i].instance == "la24"
        for i in range(12, 16):
            assert results[i].instance == "abz8"
        for i in range(16, 20):
            assert results[i].instance == "dmu21"
        for i in range(20, 24):
            assert results[i].instance == "la24"

        for i in range(12):
            assert results[i].rand_seed == results[i + 12].rand_seed

        for e in results:
            assert e.total_fes == (1 if e.algorithm == "1rs" else 300)
            assert e.max_fes == 300
            assert e.max_time_millis is None
            assert e.last_improvement_fe > 0
            assert e.last_improvement_time_millis >= 0
            assert e.best_f > 0
            assert e.goal_f > 0
            assert e.best_f > e.goal_f

        with temp_file(directory=base_dir,
                       suffix=logging.FILE_SUFFIX) as path:
            er_to_csv(results=results, file=path)

            results2: list[EndResult] = []
            results2.extend(er_from_csv(file=path))

            assert results == results2

        es_hc_a = es_create(results[12:16])
        assert es_hc_a.instance == "abz8"
        assert es_hc_a.algorithm == "hc_swap2"
        assert es_hc_a.goal_f == 648
        assert es_hc_a.best_f.minimum >= 648
        assert es_hc_a.total_fes.maximum <= 300
        assert es_hc_a.best_f_scaled.mean_geom >= 1
        assert es_hc_a.ert_fes > 0
        assert es_hc_a.ert_time_millis > 0

        if is_ci_run():
            es_rs_a = es_create(results[0:4])
            assert es_rs_a.instance == "abz8"
            assert es_rs_a.algorithm == "1rs"
            assert es_rs_a.goal_f == 648
            assert es_rs_a.best_f.minimum >= 648
            assert es_rs_a.total_fes.maximum == 1
            assert es_rs_a.best_f_scaled.mean_geom >= 1
            assert es_rs_a.ert_fes > 0
            assert es_rs_a.ert_time_millis > 0

            es_rs_l = es_create(results[8:12])
            assert es_rs_l.instance == "la24"
            assert es_rs_l.algorithm == "1rs"
            assert es_rs_l.goal_f == 935
            assert es_rs_l.best_f.minimum >= 935
            assert es_rs_l.total_fes.maximum <= 300
            assert es_rs_l.best_f_scaled.mean_geom >= 1
            assert es_rs_l.ert_fes > 0
            assert es_rs_l.ert_time_millis > 0

            es = es_create(results[12:20])
            assert es.instance is None
            assert es.algorithm == "hc_swap2"
            assert isinstance(es.goal_f, SampleStatistics)
            assert es.goal_f.minimum == 648
            assert es.goal_f.maximum == 4380
            assert es.best_f.minimum >= 648
            assert es.best_f.maximum >= 4380
            assert es.total_fes.maximum <= 300
            assert es.best_f_scaled.mean_geom >= 1
            assert es.ert_fes > 0
            assert es.ert_time_millis > 0

            es_all = es_create(results[0:24])
            assert es_all.instance is None
            assert es_all.algorithm is None
            assert isinstance(es_all.goal_f, SampleStatistics)
            assert es_all.goal_f.minimum == 648
            assert es_all.goal_f.maximum == 4380
            assert es_all.best_f.minimum >= 648
            assert es_all.best_f.maximum >= 4380
            assert es_all.total_fes.maximum <= 300
            assert es_all.best_f_scaled.mean_geom >= 1
            assert es_all.ert_fes > 0
            assert es_all.ert_time_millis > 0

            es_rs = es_create(results[0:12])
            assert es_rs.instance is None
            assert es_rs.algorithm == "1rs"
            assert isinstance(es_rs.goal_f, SampleStatistics)
            assert es_rs.goal_f.minimum == 648
            assert es_rs.goal_f.maximum == 4380
            assert es_rs.best_f.minimum >= 648
            assert es_rs.best_f.maximum >= 4380
            assert es_rs.total_fes.maximum <= 300
            assert es_rs.best_f_scaled.mean_geom >= 1
            assert es_rs.ert_fes > 0
            assert es_rs.ert_time_millis > 0

            es_l = es_create(results[8:12] + results[20:24])
            assert es_l.instance == "la24"
            assert es_l.algorithm is None
            assert es_l.goal_f == 935
            assert es_l.best_f.minimum >= 935
            assert es_l.total_fes.maximum <= 300
            assert es_l.best_f_scaled.mean_geom >= 1
            assert es_l.ert_fes > 0
            assert es_l.ert_time_millis > 0

            es_algos: list[EndStatistics] = list(es_from_end_results(
                results, join_all_instances=True))
            assert es_algos[0] == es_rs
            assert len(es_algos) == 2

            es_insts: list[EndStatistics] = list(es_from_end_results(
                results, join_all_algorithms=True))
            assert es_insts[2] == es_l
            assert len(es_insts) == 3

            es_sep: list[EndStatistics] = list(es_from_end_results(results))
            assert es_sep[3] == es_hc_a
            assert es_sep[0] == es_rs_a
            assert es_sep[2] == es_rs_l
            assert len(es_sep) == 6

            es_one: list[EndStatistics] = list(es_from_end_results(
                results, True, True))
            assert es_one == [es_all]
            assert len(es_one) == 1

        with temp_file(directory=base_dir,
                       suffix=logging.FILE_SUFFIX) as f:
            check: list[EndStatistics] = [es_hc_a]
            es_to_csv(check, f)
            check_2: list[EndStatistics] = list(es_from_csv(f))
            assert check_2 == check
            assert len(check_2) == 1

        progress_fes_raw: list[Progress] = list(pr_from_logs(
            base_dir, time_unit=bs.TIME_UNIT_FES, f_name=bs.F_NAME_RAW))
        progress_fes_raw.sort()
        assert len(progress_fes_raw) == 24
        for idx, pr in enumerate(progress_fes_raw):
            assert isinstance(pr, Progress)
            assert pr.algorithm == results[idx].algorithm
            assert pr.instance == results[idx].instance
            assert pr.rand_seed == results[idx].rand_seed
            assert pr.f[-1] == results[idx].best_f
            assert pr.time[-1] >= results[idx].last_improvement_fe
            assert pr.time[-1] <= results[idx].total_fes

        ert = ert_create(progress_fes_raw)
        assert ert.instance is None
        assert ert.algorithm is None
        assert ert.n == 24
        assert ert.f_name == bs.F_NAME_RAW
        assert ert.time_unit == bs.TIME_UNIT_FES
        assert ert.ert[0, 0] == 648
        for i, x in enumerate(ert.ert[:, 0]):
            assert compute_single_ert(progress_fes_raw, x) == ert.ert[i, 1]

        if is_ci_run():
            progress_ms_raw: list[Progress] = list(pr_from_logs(
                base_dir, time_unit=bs.TIME_UNIT_MILLIS, f_name=bs.F_NAME_RAW))
            progress_ms_raw.sort()
            assert len(progress_ms_raw) == 24
            for idx, pr in enumerate(progress_ms_raw):
                assert isinstance(pr, Progress)
                assert pr.algorithm == results[idx].algorithm
                assert pr.instance == results[idx].instance
                assert pr.rand_seed == results[idx].rand_seed
                assert pr.f[-1] == results[idx].best_f
                assert pr.time[-1] >= results[idx].last_improvement_time_millis
                assert pr.time[-1] <= results[idx].total_time_millis

            progress_fes_std: list[Progress] = list(pr_from_logs(
                base_dir, time_unit=bs.TIME_UNIT_FES, f_name=bs.F_NAME_SCALED))
            progress_fes_std.sort()
            assert len(progress_fes_std) == 24
            for idx, pr in enumerate(progress_fes_std):
                assert isinstance(pr, Progress)
                assert pr.algorithm == results[idx].algorithm
                assert pr.instance == results[idx].instance
                assert pr.rand_seed == results[idx].rand_seed
                assert np.all(pr.f >= 1)
                assert pr.time[-1] >= results[idx].last_improvement_fe
                assert pr.time[-1] <= results[idx].total_fes
                assert np.array_equal(pr.time, progress_fes_raw[idx].time)

            progress_ms_nrm: list[Progress] = list(pr_from_logs(
                base_dir, time_unit=bs.TIME_UNIT_MILLIS,
                f_name=bs.F_NAME_NORMALIZED))
            assert len(progress_ms_nrm) == 24
            progress_ms_nrm.sort()
            for idx, pr in enumerate(progress_ms_nrm):
                assert isinstance(pr, Progress)
                assert pr.algorithm == results[idx].algorithm
                assert pr.instance == results[idx].instance
                assert pr.rand_seed == results[idx].rand_seed
                assert np.all(pr.f >= 0)
                assert pr.time[-1] >= results[idx].last_improvement_time_millis
                assert pr.time[-1] <= results[idx].total_time_millis
                assert np.array_equal(pr.time, progress_ms_raw[idx].time)

        stat_names = ["min", "med", "mean", "geom", "max",
                      "mean-sd", "mean+sd", "sd",
                      "q10", "q90", "q159", "q841"]
        stat_runs: list[StatRun] = []
        sr_create(source=progress_fes_raw,
                  statistics=stat_names,
                  consumer=stat_runs.append)
        assert len(stat_runs) == len(stat_names)

        if is_ci_run():
            all_progress = []
            all_progress.extend(progress_ms_raw)
            all_progress.extend(progress_fes_raw)
            all_progress.extend(progress_fes_std)

            stat_runs.clear()
            st_from_progress(all_progress, stat_names, stat_runs.append,
                             False, False)
            assert len(stat_runs) == len(stat_names) * 3 * 3 * 2
            stat_runs.clear()
            st_from_progress(all_progress, stat_names, stat_runs.append,
                             True, False)
            assert len(stat_runs) == len(stat_names) * 3 * 3
            stat_runs.clear()
            st_from_progress(all_progress, stat_names, stat_runs.append,
                             False, True)
            assert len(stat_runs) == len(stat_names) * 3 * 2
            stat_runs.clear()
            st_from_progress(all_progress, stat_names, stat_runs.append,
                             True, True)
            assert len(stat_runs) == len(stat_names) * 3
