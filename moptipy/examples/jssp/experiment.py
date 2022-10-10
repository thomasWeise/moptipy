"""Run the moptipy example experiment."""
import os.path as pp
import sys
from typing import Tuple, Dict, Final, Iterable, Callable, \
    Optional, Union, cast, Any, List

import moptipy.api.experiment as ex
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.random_walk import RandomWalk
from moptipy.algorithms.single_random_sample import SingleRandomSample
from moptipy.algorithms.so.ea import EA
from moptipy.algorithms.so.hill_climber import HillClimber
from moptipy.algorithms.so.hill_climber_with_restarts import \
    HillClimberWithRestarts
from moptipy.algorithms.so.rls import RLS
from moptipy.api.algorithm import Algorithm
from moptipy.api.execution import Execution
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.operators.permutations.op1_swapn import Op1SwapN
from moptipy.operators.permutations.op2_gap import \
    Op2GeneralizedAlternatingPosition
from moptipy.operators.permutations.op2_ox2 import Op2OrderBased
from moptipy.algorithms.so.greedy_2plus1_ea_mod import GreedyTwoPlusOneEAmod
from moptipy.spaces.permutations import Permutations
from moptipy.utils.console import logger
from moptipy.utils.help import help_screen
from moptipy.utils.path import Path
from moptipy.utils.types import type_error

#: The default instances to be used in our experiment. These have been
#: computed via instance_selector.propose_instances.
#: The instances in this tuple are sorted by the scale of the search space,
#: if that search space is the order-based encoding, i.e., permutations with
#: repetitions. This is almost the same as an ordering by the number  of
#: possible (feasible or infeasible) Gantt charts than can be constructed, with
#: one exception: For `dmu67`, the search space size is `2.768*(10**1241)` and
#: for `dmu72`, it is `3.862*(10**1226)`, i.e., `dmu67` has a larger search
#: order-based encoding search space than `dmu72`. However, for `dmu72`, we can
#: construct `1.762*(10**967)` different (feasible or infeasible) Gantt charts,
#: whereas for `dmu67`, we can only construct `1.710*(10**958)`, i.e., fewer.
#: Therefore, `dmu67` in this ordering here comes after `dmu72`, but it would
#: come before if we were sorting instances by the solution space size.
EXPERIMENT_INSTANCES: \
    Final[Tuple[str, str, str, str, str, str, str, str]] = \
    ('orb06', 'la38', 'abz8', 'yn4', 'swv14', 'dmu72', 'dmu67', 'ta70')

#: The number of runs per instance in our JSSP experiment.
#: For each combination of algorithm setup and JSSP instance, we will
#: perform exactly this many runs.
EXPERIMENT_RUNS: Final[int] = 23

#: We will perform two minutes per run.
EXPERIMENT_RUNTIME_MS: Final[int] = 2 * 60 * 1000

#: The default set of algorithms for our experiments.
#: Each of them is a Callable that receives two parameters, the instance
#: `inst` and the permutation with repetitions-space `pwr`.
DEFAULT_ALGORITHMS: Final[List[
    Callable[[Instance, Permutations], Algorithm]]] = [
    lambda inst, pwr: SingleRandomSample(Op0Shuffle(pwr)),  # single sample
    lambda inst, pwr: RandomSampling(Op0Shuffle(pwr)),  # random sampling
    lambda inst, pwr: HillClimber(Op0Shuffle(pwr), Op1Swap2()),  # hill climb.
    lambda inst, pwr: RLS(Op0Shuffle(pwr), Op1Swap2()),  # RLS
    lambda inst, pwr: RandomWalk(Op0Shuffle(pwr), Op1Swap2()),  # random walk
    lambda inst, pwr: HillClimber(Op0Shuffle(pwr), Op1SwapN()),  # hill climb.
    lambda inst, pwr: RLS(Op0Shuffle(pwr), Op1SwapN()),  # RLS
]
for scale in range(7, 21):  # add the hill climbers with restarts
    DEFAULT_ALGORITHMS.append(cast(
        Callable[[Instance, Permutations], Algorithm],
        lambda inst, pwr, i=scale: HillClimberWithRestarts(
            Op0Shuffle(pwr), Op1Swap2(), 2 ** i))  # hill climb. with restarts
    )
    DEFAULT_ALGORITHMS.append(cast(
        Callable[[Instance, Permutations], Algorithm],
        lambda inst, pwr, i=scale: HillClimberWithRestarts(
            Op0Shuffle(pwr), Op1SwapN(), 2 ** i))  # hill climb. with restarts
    )
for muexp in range(0, 14):
    mu: int = 2 ** muexp
    for lambda_ in sorted({1, max(1, muexp), round(mu ** 0.5), mu, mu + mu,
                           4 * mu, 8 * mu}):
        if lambda_ > 65536:
            continue
        DEFAULT_ALGORITHMS.append(cast(
            Callable[[Instance, Permutations], Algorithm],
            lambda inst, pwr, mm=mu, ll=lambda_: EA(
                Op0Shuffle(pwr), Op1Swap2(), None,
                mm, ll, 0.0))  # EA without crossover
        )
        if 1 < mu < 32:
            brr = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.05, 0.25] \
                if (mu == 2) and (lambda_ <= 3) else [0.01, 0.05, 0.25]
            for br in brr:
                DEFAULT_ALGORITHMS.append(cast(
                    Callable[[Instance, Permutations], Algorithm],
                    lambda inst, pwr, mm=mu, ll=lambda_, bb=br: EA(
                        Op0Shuffle(pwr), Op1Swap2(),
                        Op2OrderBased(pwr),
                        mm, ll, bb))  # EA with crossover 1
                )
                DEFAULT_ALGORITHMS.append(cast(
                    Callable[[Instance, Permutations], Algorithm],
                    lambda inst, pwr, mm=mu, ll=lambda_, bb=br: EA(
                        Op0Shuffle(pwr), Op1Swap2(),
                        Op2GeneralizedAlternatingPosition(pwr),
                        mm, ll, bb))  # EA with crossover 2
                )

                DEFAULT_ALGORITHMS.append(cast(
                    Callable[[Instance, Permutations], Algorithm],
                    lambda inst, pwr, mm=mu, ll=lambda_, bb=br: EA(
                        Op0Shuffle(pwr), Op1Swap2(),
                        Op2GeneralizedAlternatingPosition(pwr),
                        mm, ll, bb))  # EA with crossover 2
                )

DEFAULT_ALGORITHMS.append(
    lambda inst, pwr: GreedyTwoPlusOneEAmod(
        Op0Shuffle(pwr), Op1Swap2(),
        Op2GeneralizedAlternatingPosition(pwr)))
DEFAULT_ALGORITHMS.append(
    lambda inst, pwr: GreedyTwoPlusOneEAmod(
        Op0Shuffle(pwr), Op1SwapN(),
        Op2GeneralizedAlternatingPosition(pwr)))


def run_experiment(base_dir: str = pp.join(".", "results"),
                   algorithms: Iterable[
                       Callable[[Instance, Permutations],
                                Algorithm]] = tuple(DEFAULT_ALGORITHMS),
                   instances: Iterable[str] = EXPERIMENT_INSTANCES,
                   n_runs: int = EXPERIMENT_RUNS,
                   max_time: Optional[int] = EXPERIMENT_RUNTIME_MS,
                   max_fes: Optional[int] = None,
                   n_threads: Optional[int] = None) -> None:
    """
    Run the experiment.

    :param algorithms: the algorithm factories.
        Each factory receives as input one JSSP `Instance` and one instance
        of `PermutationWithRepetition`. It returns either an instance of
        `Algorithm` or of `Execution`.
    :param instances: the JSSP instance names
    :param base_dir: the base directory
    :param n_runs: the number of runs
    :param max_time: the maximum runtime in milliseconds
    :param max_fes: the maximum runtime in FEs
    :param n_threads: the number of threads
    """
    # The initial parameter validity checks.
    if not isinstance(base_dir, str):
        raise type_error(base_dir, "base_dir", str)
    if not isinstance(algorithms, Iterable):
        raise type_error(algorithms, "algorithms", Iterable)
    if not isinstance(instances, Iterable):
        raise type_error(instances, "instances", Iterable)
    if not isinstance(n_runs, int):
        raise type_error(n_runs, "n_runs", int)
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, but is {n_runs}.")
    if max_time is not None:
        if not isinstance(max_time, int):
            raise type_error(max_time, "max_time", int)
        if max_time <= 0:
            raise ValueError(f"max_time must be positive, but is {max_time}.")
    if max_fes is not None:
        if not isinstance(max_fes, int):
            raise type_error(max_fes, "max_fes", int)
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

            :param inst: the JSSP instance
            :param algor: the algorithm creator
            :return: an Execution for the experiment
            """
            pwr: Permutations = Permutations.with_repetitions(
                inst.jobs, inst.machines)

            val: Union[Execution, Algorithm] = algor(inst, pwr)
            experiment: Execution
            if isinstance(val, Execution):
                experiment = cast(Execution, val)
            else:
                if not isinstance(val, Algorithm):
                    raise type_error(val, "result of factory function",
                                     Algorithm)
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
                               "perform_warmup": True,
                               "warmup_fes": 20,
                               "perform_pre_warmup": True,
                               "pre_warmup_fes": 20}
    if n_threads is not None:
        ikwargs["n_threads"] = n_threads

    ex.run_experiment(**ikwargs)  # invoke the actual experiment


# Execute experiment if run as script
if __name__ == '__main__':
    help_screen(
        "JSSP Experiment Executor", __file__,
        "Run the JSSP experiment.",
        [("results_dir",
          "the directory where the results should be stored",
          True),
         ("n_threads",
          "the number of threads to use for the experiment",
          True)])
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
