"""Test the execution of an experiment on the JSSP."""
import os.path as pt
from os import listdir

from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.so.hill_climber import HillClimber
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.permutations import Permutations
from moptipy.utils.temp import TempDir

instances = [lambda: Instance.from_resource("dmu01"),
             lambda: Instance.from_resource("abz7"),
             lambda: Instance.from_resource("demo")]


def algo_1(inst: Instance) -> Execution:
    """Execute the first test algorithm."""
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


def algo_2(inst: Instance) -> Execution:
    """Execute the second test algorithm."""
    ss = Permutations.with_repetitions(inst.jobs, inst.machines)
    sos = GanttSpace(inst)
    op0 = Op0Shuffle(ss)
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


def test_experiment_jssp() -> None:
    """Run the JSSP test experiment."""
    with TempDir.create() as base_dir:
        run_experiment(instances=instances,
                       setups=[algo_1, algo_2],
                       n_runs=10,
                       base_dir=base_dir,
                       n_threads=3,
                       perform_warmup=False)

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
