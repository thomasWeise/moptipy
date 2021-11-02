"""Run the moptipy example experiment."""
import os.path as pp
from typing import Tuple, Dict, Final, Iterable, Callable, \
    Optional, Union, cast, Any

import moptipy.api.experiment as ex
from moptipy.algorithms.ea1p1 import EA1p1
from moptipy.algorithms.hill_climber import HillClimber
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.random_walk import RandomWalk
from moptipy.algorithms.single_random_sample import SingleRandomSample
from moptipy.api.algorithm import Algorithm
from moptipy.api.execution import Execution
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.operators.pwr.op1_swap2 import Op1Swap2
from moptipy.spaces.permutationswr import PermutationsWithRepetitions

#: The default instances to be used in our experiment. These have been
#: computed via instance_selector.propose_instances.
EXPERIMENT_INSTANCES: \
    Final[Tuple[str, str, str, str, str, str, str, str]] = \
    ('abz8', 'dmu40', 'ft06', 'la09', 'swv18', 'ta54', 'ta79', 'yn2')

#: The number of runs per instance in our experiment.
EXPERIMENT_RUNS: Final[int] = 7

#: We will perform two minutes per run.
EXPERIMENT_RUNTIME_MS: Final[int] = 2 * 60 * 1000


#: The default set of algorithms for our experiments
DEFAULT_ALGORITHMS: Final[Tuple[Callable, ...]] = (
    lambda inst: EA1p1(Op0Shuffle(), Op1Swap2()),  # (1+1)-EA
    lambda inst: HillClimber(Op0Shuffle(), Op1Swap2()),  # hill climber
    lambda inst: RandomSampling(Op0Shuffle()),  # random sampling
    lambda inst: SingleRandomSample(Op0Shuffle()),  # single random sample
    lambda inst: RandomWalk(Op0Shuffle(), Op1Swap2())  # random walk
)


def run_experiment(base_dir: str = pp.join(".", "results"),
                   algorithms: Iterable[Callable] = DEFAULT_ALGORITHMS,
                   instances: Iterable[str] = EXPERIMENT_INSTANCES,
                   n_runs: int = EXPERIMENT_RUNS,
                   max_time: Optional[int] = EXPERIMENT_RUNTIME_MS,
                   max_fes: Optional[int] = None,
                   n_threads: Optional[int] = None) -> None:
    """
    Run the experiment.

    :param Iterable[Callable] algorithms: the algorithm factories
    :param Iterable[Callable] instances: the JSSP instance names
    :param str base_dir: the base directory
    :param int n_runs: the number of runs
    :param Optional[int] max_time: the maximum runtime in milliseconds
    :param Optional[int] max_fes: the maximum runtime in FEs
    :param Optional[int] n_threads: the number of threads
    """
    if not isinstance(base_dir, str):
        raise TypeError(
            f"base_dir must be str, but is {type(base_dir)}.")
    if not isinstance(algorithms, Iterable):
        raise TypeError(
            f"algorithms must be iterable, but is {type(algorithms)}.")
    if not isinstance(instances, Iterable):
        raise TypeError(
            f"Instances must be iterable, but is {type(instances)}.")
    if not isinstance(n_runs, int):
        raise TypeError(f"n_runs must be int but is {type(n_runs)}.")
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, but is {n_runs}.")
    if max_time is not None:
        if not isinstance(max_time, int):
            raise TypeError(f"max_time must be int but is {type(max_time)}.")
        if max_time <= 0:
            raise ValueError(f"max_time must be positive, but is {max_time}.")
    if max_fes is not None:
        if not isinstance(max_fes, int):
            raise TypeError(f"max_fes must be int but is {type(max_fes)}.")
        if max_time <= 0:
            raise ValueError(f"max_fes must be positive, but is {max_fes}.")
    if (max_fes is None) and (max_time is None):
        raise ValueError("Either max_fes or max_time must be provided.")

    inst_gens = [(lambda s=s: Instance.from_resource(s)) for s in instances]
    if len(inst_gens) <= 0:
        raise ValueError("There must be at least one instance.")
    algo_gens = []
    for algo in algorithms:
        def creator(inst: Instance, algor=algo):
            val: Union[Execution, Algorithm] = algor(inst)
            experiment: Execution
            if isinstance(val, Execution):
                experiment = cast(Execution, val)
            else:
                experiment = Execution()
                experiment.set_algorithm(cast(Algorithm, val))
            if max_time is not None:
                experiment.set_max_time_millis(max_time)
            if max_fes is not None:
                experiment.set_max_fes(max_fes)
            experiment.set_objective(Makespan(inst))
            experiment.set_solution_space(GanttSpace(inst))
            experiment.set_search_space(
                PermutationsWithRepetitions(inst.jobs, inst.machines))
            experiment.set_encoding(OperationBasedEncoding(inst))
            experiment.set_log_improvements()
            return experiment

        algo_gens.append(creator)
        del creator
    if len(algo_gens) <= 0:
        raise ValueError("There must be at least one algorithm.")

    kwargs: Dict[str, Any] = {"base_dir": base_dir,
                              "instances": inst_gens,
                              "setups": algo_gens,
                              "n_runs": n_runs,
                              "perform_warmup": True}
    if n_threads is not None:
        kwargs["n_threads"] = n_threads

    ex.run_experiment(**kwargs)


# Execute experiment if run as script
if __name__ == '__main__':
    run_experiment()
