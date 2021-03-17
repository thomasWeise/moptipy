"""Test the execution of an experiment and parsing the log files the JSSP."""

from typing import List

import moptipy.operators.pwr as pwr
from moptipy.algorithms import HillClimber, RandomSampling
from moptipy.api import Execution, run_experiment
from moptipy.evaluation import EndResult, parse_logs, end_results_to_csv, \
    csv_to_end_results, EndStatistics
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
        parse_logs(base_dir, results)

        assert len(results) == (4 * 2 * 3)
        results.sort()

        for i in range(0, 4 * 3):
            assert results[i].algorithm == "hc_swap2"
        for i in range(4 * 3, len(results)):
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
            end_results_to_csv(results=results, file=path)

            results2: List[EndResult] = list()
            csv_to_end_results(file=path, collector=results2)

            assert results == results2

        es = EndStatistics.create(results[0:4])
        assert es.instance == "abz8"
        assert es.algorithm == "hc_swap2"
        assert es.goal_f == 648
        assert es.best_f.minimum >= 648
        assert es.total_fes.maximum <= 300
        assert es.best_f_scaled.mean_geom >= 1
        assert es.ert_fes > 0
        assert es.ert_time_millis > 0

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

        es = EndStatistics.create(results[0:24])
        assert es.instance is None
        assert es.algorithm is None
        assert es.goal_f.minimum == 648
        assert es.goal_f.maximum == 4380
        assert es.best_f.minimum >= 648
        assert es.best_f.maximum >= 4380
        assert es.total_fes.maximum <= 300
        assert es.best_f_scaled.mean_geom >= 1
        assert es.ert_fes > 0
        assert es.ert_time_millis > 0
