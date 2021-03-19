"""Test the execution of an experiment and parsing the log files the JSSP."""

from typing import List

import moptipy.operators.pwr as pwr
from moptipy.algorithms import HillClimber, RandomSampling
from moptipy.api import Execution, run_experiment
from moptipy.evaluation import EndResult, EndStatistics
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
    algo = RandomSampling(op0)
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
            assert results[i].algorithm == "hc_swap2"
        for i in range(12, 24):
            assert results[i].algorithm == "rs"

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
            assert e.total_fes == 300
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

        es_hc_a = EndStatistics.create(results[0:4])
        assert es_hc_a.instance == "abz8"
        assert es_hc_a.algorithm == "hc_swap2"
        assert es_hc_a.goal_f == 648
        assert es_hc_a.best_f.minimum >= 648
        assert es_hc_a.total_fes.maximum <= 300
        assert es_hc_a.best_f_scaled.mean_geom >= 1
        assert es_hc_a.ert_fes > 0
        assert es_hc_a.ert_time_millis > 0

        es_hc_d = EndStatistics.create(results[4:8])
        assert es_hc_d.instance == "dmu21"
        assert es_hc_d.algorithm == "hc_swap2"
        assert es_hc_d.goal_f == 4380
        assert es_hc_d.best_f.minimum >= 4380
        assert es_hc_d.total_fes.maximum <= 300
        assert es_hc_d.best_f_scaled.mean_geom >= 1
        assert es_hc_d.ert_fes > 0
        assert es_hc_d.ert_time_millis > 0

        es_hc_l = EndStatistics.create(results[8:12])
        assert es_hc_l.instance == "la24"
        assert es_hc_l.algorithm == "hc_swap2"
        assert es_hc_l.goal_f == 935
        assert es_hc_l.best_f.minimum >= 935
        assert es_hc_l.total_fes.maximum <= 300
        assert es_hc_l.best_f_scaled.mean_geom >= 1
        assert es_hc_l.ert_fes > 0
        assert es_hc_l.ert_time_millis > 0

        es_rs_a = EndStatistics.create(results[12:16])
        assert es_rs_a.instance == "abz8"
        assert es_rs_a.algorithm == "rs"
        assert es_rs_a.goal_f == 648
        assert es_rs_a.best_f.minimum >= 648
        assert es_rs_a.total_fes.maximum <= 300
        assert es_rs_a.best_f_scaled.mean_geom >= 1
        assert es_rs_a.ert_fes > 0
        assert es_rs_a.ert_time_millis > 0

        es_rs_d = EndStatistics.create(results[16:20])
        assert es_rs_d.instance == "dmu21"
        assert es_rs_d.algorithm == "rs"
        assert es_rs_d.goal_f == 4380
        assert es_rs_d.best_f.minimum >= 4380
        assert es_rs_d.total_fes.maximum <= 300
        assert es_rs_d.best_f_scaled.mean_geom >= 1
        assert es_rs_d.ert_fes > 0
        assert es_rs_d.ert_time_millis > 0

        es_rs_l = EndStatistics.create(results[20:24])
        assert es_rs_l.instance == "la24"
        assert es_rs_l.algorithm == "rs"
        assert es_rs_l.goal_f == 935
        assert es_rs_l.best_f.minimum >= 935
        assert es_rs_l.total_fes.maximum <= 300
        assert es_rs_l.best_f_scaled.mean_geom >= 1
        assert es_rs_l.ert_fes > 0
        assert es_rs_l.ert_time_millis > 0

        es = EndStatistics.create(results[0:8])
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

        es_hc = EndStatistics.create(results[0:12])
        assert es_hc.instance is None
        assert es_hc.algorithm == "hc_swap2"
        assert es_hc.goal_f.minimum == 648
        assert es_hc.goal_f.maximum == 4380
        assert es_hc.best_f.minimum >= 648
        assert es_hc.best_f.maximum >= 4380
        assert es_hc.total_fes.maximum <= 300
        assert es_hc.best_f_scaled.mean_geom >= 1
        assert es_hc.ert_fes > 0
        assert es_hc.ert_time_millis > 0

        es_rs = EndStatistics.create(results[12:24])
        assert es_rs.instance is None
        assert es_rs.algorithm == "rs"
        assert es_rs.goal_f.minimum == 648
        assert es_rs.goal_f.maximum == 4380
        assert es_rs.best_f.minimum >= 648
        assert es_rs.best_f.maximum >= 4380
        assert es_rs.total_fes.maximum <= 300
        assert es_rs.best_f_scaled.mean_geom >= 1
        assert es_rs.ert_fes > 0
        assert es_rs.ert_time_millis > 0

        es_a = EndStatistics.create(results[0:4] + results[12:16])
        assert es_a.instance == "abz8"
        assert es_a.algorithm is None
        assert es_a.goal_f == 648
        assert es_a.best_f.minimum >= 648
        assert es_a.total_fes.maximum <= 300
        assert es_a.best_f_scaled.mean_geom >= 1
        assert es_a.ert_fes > 0
        assert es_a.ert_time_millis > 0

        es_d = EndStatistics.create(results[4:8] + results[16:20])
        assert es_d.instance == "dmu21"
        assert es_d.algorithm is None
        assert es_d.goal_f == 4380
        assert es_d.best_f.minimum >= 4380
        assert es_d.total_fes.maximum <= 300
        assert es_d.best_f_scaled.mean_geom >= 1
        assert es_d.ert_fes > 0
        assert es_d.ert_time_millis > 0

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
        assert es_algos == [es_hc, es_rs]
        assert len(es_algos) == 2

        es_insts = list()
        EndStatistics.from_end_results(results, es_insts,
                                       join_all_algorithms=True)
        assert es_insts == [es_a, es_d, es_l]
        assert len(es_insts) == 3

        es_sep = list()
        EndStatistics.from_end_results(results, es_sep)
        assert es_sep == [es_hc_a, es_hc_d, es_hc_l,
                          es_rs_a, es_rs_d, es_rs_l]
        assert len(es_sep) == 6

        es_one = list()
        EndStatistics.from_end_results(results, es_one,
                                       True, True)
        assert es_one == [es_all]
        assert len(es_one) == 1

        with TempFile(directory=base_dir,
                      suffix=logging.FILE_SUFFIX) as csv:
            f = str(csv)

            EndStatistics.to_csv(es_algos, f)
            es_algos2 = list()
            EndStatistics.from_csv(f, es_algos2)
            assert es_algos2 == es_algos
            assert len(es_algos2) == 2

            EndStatistics.to_csv(es_insts, f)
            es_insts2 = list()
            EndStatistics.from_csv(f, es_insts2)
            assert es_insts2 == es_insts
            assert len(es_insts2) == 3

            EndStatistics.to_csv(es_sep, f)
            es_sep2 = list()
            EndStatistics.from_csv(f, es_sep2)
            assert es_sep2 == es_sep
            assert len(es_sep2) == 6

            EndStatistics.to_csv(es_one, f)
            es_one2 = list()
            EndStatistics.from_csv(f, es_one2)
            assert es_one2 == es_one
            assert len(es_one2) == 1
