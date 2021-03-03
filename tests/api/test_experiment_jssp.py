import os.path as pt
from os import listdir

import moptipy.operators.pwr as pwr
from moptipy.algorithms import HillClimber, RandomSampling
from moptipy.api import Experiment, run_experiment
from moptipy.examples.jssp import JSSPInstance, Makespan,\
    OperationBasedEncoding, GanttSpace
from moptipy.spaces import PermutationsWithRepetitions
from moptipy.utils.io import TempDir

instances = [lambda: JSSPInstance.from_resource("dmu01"),
             lambda: JSSPInstance.from_resource("abz7"),
             lambda: JSSPInstance.from_resource("demo")]


def algo_1(inst) -> Experiment:
    ss = PermutationsWithRepetitions(inst.jobs, inst.machines)
    sos = GanttSpace(inst)
    op0 = pwr.Op0()
    op1 = pwr.Op1Swap2()
    algo = HillClimber(op0, op1)
    ex = Experiment()
    ex.set_algorithm(algo)
    ex.set_search_space(ss)
    ex.set_solution_space(sos)
    ex.set_encoding(OperationBasedEncoding(inst))
    ex.set_objective(Makespan(inst))
    ex.set_max_fes(1000)
    ex.set_log_improvements()
    return ex


def algo_2(inst) -> Experiment:
    ss = PermutationsWithRepetitions(inst.jobs, inst.machines)
    sos = GanttSpace(inst)
    op0 = pwr.Op0()
    algo = RandomSampling(op0)
    ex = Experiment()
    ex.set_algorithm(algo)
    ex.set_search_space(ss)
    ex.set_solution_space(sos)
    ex.set_encoding(OperationBasedEncoding(inst))
    ex.set_objective(Makespan(inst))
    ex.set_max_fes(1000)
    ex.set_log_improvements()
    return ex


def test_experiment_jssp():
    with TempDir() as td:
        base_dir = str(td)

        run_experiment(instances=instances,
                       setups=[algo_1, algo_2],
                       n_runs=10,
                       base_dir=base_dir,
                       n_threads=3)

        algos = listdir(base_dir)
        assert len(algos) == 2
        algos.sort()
        assert algos == ["hc_swap2", "rs"]
        for a in algos:
            ap = pt.join(base_dir, a)
            assert pt.isdir(ap)
            insts = listdir(ap)
            assert len(insts) == 3
            insts.sort()
            assert insts == ["abz7", "demo", "dmu01"]
            for i in insts:
                ip = pt.join(ap, i)
                assert pt.isdir(ip)
                runs = listdir(ip)
                assert len(runs) == 10
                start = a + "_" + i + "_"
                end = ".txt"
                for run in runs:
                    assert run.startswith(start)
                    assert run.endswith(end)
                    rp = pt.join(ip, run)
                    assert pt.isfile(rp)
                    assert pt.getsize(rp) > 10

        run_experiment(instances=instances,
                       setups=[algo_1, algo_2],
                       n_runs=10,
                       base_dir=base_dir)

        algos = listdir(base_dir)
        assert len(algos) == 2
        algos.sort()
        assert algos == ["hc_swap2", "rs"]
        for a in algos:
            ap = pt.join(base_dir, a)
            assert pt.isdir(ap)
            insts = listdir(ap)
            assert len(insts) == 3
            insts.sort()
            assert insts == ["abz7", "demo", "dmu01"]
            for i in insts:
                ip = pt.join(ap, i)
                assert pt.isdir(ip)
                runs = listdir(ip)
                assert len(runs) == 10
                start = a + "_" + i + "_"
                end = ".txt"
                for run in runs:
                    assert run.startswith(start)
                    assert run.endswith(end)
                    rp = pt.join(ip, run)
                    assert pt.isfile(rp)
                    assert pt.getsize(rp) > 10
