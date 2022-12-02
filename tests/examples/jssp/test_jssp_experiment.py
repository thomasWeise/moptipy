"""Test the demo functions."""
from moptipy.api.algorithm import Algorithm
from moptipy.api.execution import Execution
from moptipy.examples.jssp.experiment import ALGORITHMS, INSTANCES
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.spaces.permutations import Permutations


def test_default_algorithms_on_default_instances() -> None:
    """Test the default algorithms on the default instances."""
    for inst_name in INSTANCES:
        inst = Instance.from_resource(inst_name)
        assert isinstance(inst, Instance)
        perm = Permutations.with_repetitions(inst.jobs, inst.machines)
        assert isinstance(perm, Permutations)
        sol = GanttSpace(inst)
        assert isinstance(sol, GanttSpace)
        gpm = OperationBasedEncoding(inst)
        assert isinstance(gpm, OperationBasedEncoding)
        makespan = Makespan(inst)
        assert isinstance(makespan, Makespan)
        for algo_factory in ALGORITHMS:
            algo = algo_factory(inst, perm)
            assert isinstance(algo, Algorithm)
            exe = Execution()
            exe.set_algorithm(algo)
            exe.set_search_space(perm)
            exe.set_solution_space(sol)
            exe.set_encoding(gpm)
            exe.set_objective(makespan)
            exe.set_max_fes(64)
            with exe.execute() as proc:
                assert proc.has_best()
