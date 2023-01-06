"""Test converting the results of an experiment to the IOHprofiler format."""

from moptipy.algorithms.single_random_sample import SingleRandomSample
from moptipy.algorithms.so.ea import EA
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.ioh_analyzer import moptipy_to_ioh_analyzer
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.permutations import Permutations
from moptipy.utils.temp import TempDir

inst_names = ["dmu21", "abz8", "la24"]
instances = [lambda n=ins: Instance.from_resource(n)
             for ins in inst_names]


def algo_1(inst) -> Execution:
    """The first algorithm."""
    ss = Permutations.with_repetitions(inst.jobs, inst.machines)
    sos = GanttSpace(inst)
    op0 = Op0Shuffle(ss)
    op1 = Op1Swap2()
    algo = EA(op0, op1, mu=5)
    ex = Execution()
    ex.set_algorithm(algo)
    ex.set_search_space(ss)
    ex.set_solution_space(sos)
    ex.set_encoding(OperationBasedEncoding(inst))
    ex.set_objective(Makespan(inst))
    ex.set_max_fes(50)
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
    ex.set_max_fes(50)
    ex.set_log_improvements()
    return ex


def test_experiment_jssp_to_ioh() -> None:
    """Run a quick experiment and convert to IOH format."""
    with TempDir.create() as results_dir, TempDir.create() as ioh_dir:
        run_experiment(instances=instances,
                       setups=[algo_1, algo_2],
                       n_runs=4,
                       base_dir=results_dir)
        moptipy_to_ioh_analyzer(results_dir, ioh_dir)

        algo_names = ["1rs", "ea_5_1_swap2"]
        for an in algo_names:
            adir = ioh_dir.resolve_inside(an)
            adir.enforce_dir()
            for ins in inst_names:
                info = adir.resolve_inside(f"IOHprofiler_f{ins}.info")
                info.enforce_file()
                data = adir.resolve_inside(f"data_f{ins}")
                data.enforce_dir()
                file = data.resolve_inside(f"IOHprofiler_f{ins}_DIM1.dat")
                file.enforce_file()
