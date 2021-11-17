"""Run the moptipy example experiment."""
import os.path as pp
import sys
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
from moptipy.utils.logging import logger
from moptipy.utils.path import Path

#: The default instances to be used in our experiment. These have been
#: computed via instance_selector.propose_instances.
#: The instances in this tuple are sorted by the scale, i.e., the number
#: of possible (feasible or infeasible) Gantt charts than can be constructed
#: for them. For the smallest instance, ft06, we can construct over 10**17.
#: For the largest one (ta79), more than 10**3159 are possible.
EXPERIMENT_INSTANCES: \
    Final[Tuple[str, str, str, str, str, str, str, str]] = \
    ('ft06', 'la09', 'abz8', 'yn2', 'swv18', 'ta54', 'dmu40', 'ta79')

#: The number of runs per instance in our experiment.
EXPERIMENT_RUNS: Final[int] = 11

#: We will perform two minutes per run.
EXPERIMENT_RUNTIME_MS: Final[int] = 2 * 60 * 1000


#: The default set of algorithms for our experiments.
#: Each of them is a Callable that receives two parameters, the instance
#: `inst` and the permutation with repetitions-space `pwr`.
DEFAULT_ALGORITHMS: Final[Tuple[Callable, ...]] = (
    lambda inst, pwr: EA1p1(Op0Shuffle(pwr), Op1Swap2()),  # (1+1)-EA
    lambda inst, pwr: HillClimber(Op0Shuffle(pwr), Op1Swap2()),  # hill climb.
    lambda inst, pwr: RandomSampling(Op0Shuffle(pwr)),  # random sampling
    lambda inst, pwr: SingleRandomSample(Op0Shuffle(pwr)),  # single sample
    lambda inst, pwr: RandomWalk(Op0Shuffle(pwr), Op1Swap2())  # random walk
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

    :param Iterable[Callable] algorithms: the algorithm factories.
        Each factory receives as input one JSSP `Instance` and one instance
        of `PermutationWithRepetition`. It returns either an instance of
        `Algorithm` or of `Execution`.
    :param Iterable[Callable] instances: the JSSP instance names
    :param str base_dir: the base directory
    :param int n_runs: the number of runs
    :param Optional[int] max_time: the maximum runtime in milliseconds
    :param Optional[int] max_fes: the maximum runtime in FEs
    :param Optional[int] n_threads: the number of threads
    """
    # The intitial parameter validity checks.
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
    del instances

    # In this loop, we convert the simple algorithm lambdas to execution
    # creators.
    algo_gens = []
    for algo in algorithms:
        # we create one constructor for each algorithm factory
        def creator(inst: Instance, algor: Callable = algo) -> Execution:
            """
            Create the algorithm for a given instance.

            :param Instance inst: the JSSP instance
            :param Callable algor: the algorithm creator
            :return: an Execution for the experiment
            :rtype: Execution
            """
            pwr: PermutationsWithRepetitions = \
                PermutationsWithRepetitions(inst.jobs, inst.machines)

            val: Union[Execution, Algorithm] = algor(inst, pwr)
            experiment: Execution
            if isinstance(val, Execution):
                experiment = cast(Execution, val)
            else:
                if not isinstance(val, Algorithm):
                    raise TypeError("Factory must return Algorithm or "
                                    f"Execution, but returns {type(val)}.")
                experiment = Execution()
                experiment.set_algorithm(cast(Algorithm, val))

            if max_time is not None:
                experiment.set_max_time_millis(max_time)
            if max_fes is not None:
                experiment.set_max_fes(max_fes)
            experiment.set_objective(Makespan(inst))
            experiment.set_solution_space(GanttSpace(inst))
            experiment.set_search_space(pwr)
            experiment.set_encoding(OperationBasedEncoding(inst))
            experiment.set_log_improvements()
            return experiment

        algo_gens.append(creator)  # add constructor to generator list
        del creator  # the creator is no longer needed
    del algorithms

    if len(algo_gens) <= 0:
        raise ValueError("There must be at least one algorithm.")

    ikwargs: Dict[str, Any] = {"base_dir": base_dir,
                               "instances": inst_gens,
                               "setups": algo_gens,
                               "n_runs": n_runs,
                               "perform_warmup": True}
    if n_threads is not None:
        ikwargs["n_threads"] = n_threads

    ex.run_experiment(**ikwargs)  # invoke the actual experiment


# Execute experiment if run as script
if __name__ == '__main__':
    mkwargs: Dict[str, Any] = {}
    if len(sys.argv) > 1:
        dest_dir: Final[Path] = Path.path(sys.argv[1])
        dest_dir.ensure_dir_exists()
        mkwargs["base_dir"] = dest_dir
        logger(f"Set base_dir '{dest_dir}'.")
        if len(sys.argv) > 2:
            n_cpu = int(sys.argv[2])
            mkwargs["n_threads"] = n_cpu
            logger(f"Set n_threads to {n_cpu}.")

    run_experiment(**mkwargs)  # invoke the experiment execution
