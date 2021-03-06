"""Test the execution of an experiment and parsing the log files the JSSP."""

from os import environ
from typing import List

import numpy as np

import moptipy.evaluation.base as bs
import moptipy.operators.pwr as pwr
from moptipy.algorithms import HillClimber, SingleRandomSample
from moptipy.api import Execution, run_experiment
from moptipy.evaluation import EndResult, EndStatistics, Progress, StatRun, \
    Ert, compute_single_ert
from moptipy.examples.jssp import Instance, Makespan, \
    OperationBasedEncoding, GanttSpace
from moptipy.spaces import PermutationsWithRepetitions
from moptipy.utils import logging
from moptipy.utils.io import TempDir, TempFile

instances = [lambda: Instance.from_resource("dmu21"),
             lambda: Instance.from_resource("abz8"),
             lambda: Instance.from_resource("la24")]


def algo_1(inst) -> Execution:
    ss = PermutationsWithRepetitions(inst.jobs, inst.machines)
    sos = GanttSpace(inst)
    op0 = pwr.Op0Shuffle()
    op1 = pwr.Op1Swap2()
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
    ss = PermutationsWithRepetitions(inst.jobs, inst.machines)
    sos = GanttSpace(inst)
    op0 = pwr.Op0Shuffle()
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


def test_experiment_jssp():
    with TempDir() as td:
        base_dir = str(td)

        run_experiment(instances=instances,
                       setups=[algo_1, algo_2],
                       n_runs=4,
                       base_dir=base_dir)

        results: List[EndResult] = list()
        EndResult.from_logs(base_dir, results)

        assert len(results) == (4 * 2 * 3)
        results.sort()

        for i in range(0, 12):
            assert results[i].algorithm == "1rs"
        for i in range(12, 24):
            assert results[i].algorithm == "hc_swap2"

        for i in range(0, 4):
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

        for i in range(0, 12):
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

        with TempFile(directory=base_dir,
                      suffix=logging.FILE_SUFFIX) as csv:
            path = str(csv)
            EndResult.to_csv(results=results, file=path)

            results2: List[EndResult] = list()
            EndResult.from_csv(file=path, collector=results2)

            assert results == results2

        es_hc_a = EndStatistics.create(results[12:16])
        assert es_hc_a.instance == "abz8"
        assert es_hc_a.algorithm == "hc_swap2"
        assert es_hc_a.goal_f == 648
        assert es_hc_a.best_f.minimum >= 648
        assert es_hc_a.total_fes.maximum <= 300
        assert es_hc_a.best_f_scaled.mean_geom >= 1
        assert es_hc_a.ert_fes > 0
        assert es_hc_a.ert_time_millis > 0

        if "GITHUB_JOB" in environ:

            es_rs_a = EndStatistics.create(results[0:4])
            assert es_rs_a.instance == "abz8"
            assert es_rs_a.algorithm == "1rs"
            assert es_rs_a.goal_f == 648
            assert es_rs_a.best_f.minimum >= 648
            assert es_rs_a.total_fes.maximum == 1
            assert es_rs_a.best_f_scaled.mean_geom >= 1
            assert es_rs_a.ert_fes > 0
            assert es_rs_a.ert_time_millis > 0

            es_rs_l = EndStatistics.create(results[8:12])
            assert es_rs_l.instance == "la24"
            assert es_rs_l.algorithm == "1rs"
            assert es_rs_l.goal_f == 935
            assert es_rs_l.best_f.minimum >= 935
            assert es_rs_l.total_fes.maximum <= 300
            assert es_rs_l.best_f_scaled.mean_geom >= 1
            assert es_rs_l.ert_fes > 0
            assert es_rs_l.ert_time_millis > 0

            es = EndStatistics.create(results[12:20])
            assert es.instance is None
            assert es.algorithm == "hc_swap2"
            assert es.goal_f.minimum == 648
            assert es.goal_f.maximum == 4380
            assert es.best_f.minimum >= 648
            assert es.best_f.maximum >= 4380
            assert es.total_fes.maximum <= 300
            assert es.best_f_scaled.mean_geom >= 1
            assert es.ert_fes > 0
            assert es.ert_time_millis > 0

            es_all = EndStatistics.create(results[0:24])
            assert es_all.instance is None
            assert es_all.algorithm is None
            assert es_all.goal_f.minimum == 648
            assert es_all.goal_f.maximum == 4380
            assert es_all.best_f.minimum >= 648
            assert es_all.best_f.maximum >= 4380
            assert es_all.total_fes.maximum <= 300
            assert es_all.best_f_scaled.mean_geom >= 1
            assert es_all.ert_fes > 0
            assert es_all.ert_time_millis > 0

            es_rs = EndStatistics.create(results[0:12])
            assert es_rs.instance is None
            assert es_rs.algorithm == "1rs"
            assert es_rs.goal_f.minimum == 648
            assert es_rs.goal_f.maximum == 4380
            assert es_rs.best_f.minimum >= 648
            assert es_rs.best_f.maximum >= 4380
            assert es_rs.total_fes.maximum <= 300
            assert es_rs.best_f_scaled.mean_geom >= 1
            assert es_rs.ert_fes > 0
            assert es_rs.ert_time_millis > 0

            es_l = EndStatistics.create(results[8:12] + results[20:24])
            assert es_l.instance == "la24"
            assert es_l.algorithm is None
            assert es_l.goal_f == 935
            assert es_l.best_f.minimum >= 935
            assert es_l.total_fes.maximum <= 300
            assert es_l.best_f_scaled.mean_geom >= 1
            assert es_l.ert_fes > 0
            assert es_l.ert_time_millis > 0

            es_algos = list()
            EndStatistics.from_end_results(results, es_algos,
                                           join_all_instances=True)
            assert es_algos[0] == es_rs
            assert len(es_algos) == 2

            es_insts = list()
            EndStatistics.from_end_results(results, es_insts,
                                           join_all_algorithms=True)
            assert es_insts[2] == es_l
            assert len(es_insts) == 3

            es_sep = list()
            EndStatistics.from_end_results(results, es_sep)
            assert es_sep[3] == es_hc_a
            assert es_sep[0] == es_rs_a
            assert es_sep[2] == es_rs_l
            assert len(es_sep) == 6

            es_one = list()
            EndStatistics.from_end_results(results, es_one,
                                           True, True)
            assert es_one == [es_all]
            assert len(es_one) == 1

        with TempFile(directory=base_dir,
                      suffix=logging.FILE_SUFFIX) as csv:
            f = str(csv)

            check = [es_hc_a]
            EndStatistics.to_csv(check, f)
            check_2 = list()
            EndStatistics.from_csv(f, check_2)
            assert check_2 == check
            assert len(check_2) == 1

        progress_fes_raw = list()
        progress_ms_raw = list()
        progress_fes_std = list()
        progress_ms_nrm = list()
        Progress.from_logs(base_dir, progress_fes_raw,
                           time_unit=bs.TIME_UNIT_FES,
                           f_name=bs.F_NAME_RAW)
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

        ert = Ert.create(progress_fes_raw)
        assert ert.instance is None
        assert ert.algorithm is None
        assert ert.n == 24
        assert ert.f_name == bs.F_NAME_RAW
        assert ert.time_unit == bs.TIME_UNIT_FES
        assert ert.ert[0, 0] == 648
        for i, x in enumerate(ert.ert[:, 0]):
            assert compute_single_ert(progress_fes_raw, x) == ert.ert[i, 1]

        if "GITHUB_JOB" in environ:

            Progress.from_logs(base_dir, progress_ms_raw,
                               time_unit=bs.TIME_UNIT_MILLIS,
                               f_name=bs.F_NAME_RAW)
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

            Progress.from_logs(base_dir, progress_fes_std,
                               time_unit=bs.TIME_UNIT_FES,
                               f_name=bs.F_NAME_SCALED)
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

            Progress.from_logs(base_dir, progress_ms_nrm,
                               time_unit=bs.TIME_UNIT_MILLIS,
                               f_name=bs.F_NAME_NORMALIZED)
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
        stat_runs = list()
        StatRun.create(source=progress_fes_raw,
                       statistics=stat_names,
                       collector=stat_runs)
        assert len(stat_runs) == len(stat_names)

        if "GITHUB_JOB" in environ:
            all_progress = list()
            all_progress.extend(progress_ms_raw)
            all_progress.extend(progress_fes_raw)
            all_progress.extend(progress_fes_std)

            stat_runs.clear()
            StatRun.from_progress(all_progress,
                                  stat_names,
                                  stat_runs,
                                  False, False)
            assert len(stat_runs) == len(stat_names) * 3 * 3 * 2
            stat_runs.clear()
            StatRun.from_progress(all_progress,
                                  stat_names,
                                  stat_runs,
                                  True, False)
            assert len(stat_runs) == len(stat_names) * 3 * 3
            stat_runs.clear()
            StatRun.from_progress(all_progress,
                                  stat_names,
                                  stat_runs,
                                  False, True)
            assert len(stat_runs) == len(stat_names) * 3 * 2
            stat_runs.clear()
            StatRun.from_progress(all_progress,
                                  stat_names,
                                  stat_runs,
                                  True, True)
            assert len(stat_runs) == len(stat_names) * 3
