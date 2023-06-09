"""Run the moptipy example experiment."""
import argparse
import os.path as pp
from typing import Any, Callable, Final, Iterable, cast

import moptipy.api.experiment as ex
from moptipy.algorithms.modules.selections.fitness_proportionate_sus import (
    FitnessProportionateSUS,
)
from moptipy.algorithms.modules.selections.tournament_with_repl import (
    TournamentWithReplacement,
)
from moptipy.algorithms.modules.selections.tournament_without_repl import (
    TournamentWithoutReplacement,
)
from moptipy.algorithms.modules.temperature_schedule import ExponentialSchedule
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.random_walk import RandomWalk
from moptipy.algorithms.single_random_sample import SingleRandomSample
from moptipy.algorithms.so.ea import EA
from moptipy.algorithms.so.fitnesses.direct import Direct
from moptipy.algorithms.so.fitnesses.rank import Rank
from moptipy.algorithms.so.general_ea import GeneralEA
from moptipy.algorithms.so.hill_climber import HillClimber
from moptipy.algorithms.so.hill_climber_with_restarts import (
    HillClimberWithRestarts,
)
from moptipy.algorithms.so.ma import MA
from moptipy.algorithms.so.marls import MARLS
from moptipy.algorithms.so.ppa import PPA
from moptipy.algorithms.so.rls import RLS
from moptipy.algorithms.so.simulated_annealing import SimulatedAnnealing
from moptipy.api.algorithm import Algorithm
from moptipy.api.execution import Execution
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.op0_forward import Op0Forward
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.operators.permutations.op1_swap_try_n import Op1SwapTryN
from moptipy.operators.permutations.op1_swapn import Op1SwapN
from moptipy.operators.permutations.op2_gap import (
    Op2GeneralizedAlternatingPosition,
)
from moptipy.spaces.permutations import Permutations
from moptipy.utils.help import argparser
from moptipy.utils.path import Path
from moptipy.utils.types import check_int_range, type_error

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
INSTANCES: \
    Final[tuple[str, str, str, str, str, str, str, str]] = \
    ("orb06", "la38", "abz8", "yn4", "swv14", "dmu72", "dmu67", "ta70")

#: The number of runs per instance in our JSSP experiment.
#: For each combination of algorithm setup and JSSP instance, we will
#: perform exactly this many runs.
EXPERIMENT_RUNS: Final[int] = 23

#: We will perform two minutes per run.
EXPERIMENT_RUNTIME_MS: Final[int] = 2 * 60 * 1000

#: The default set of algorithms for our experiments.
#: Each of them is a Callable that receives two parameters, the instance
#: `inst` and the permutation with repetitions-space `pwr`.
ALGORITHMS: Final[list[
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
    ALGORITHMS.append(cast(
        Callable[[Instance, Permutations], Algorithm],
        lambda inst, pwr, i=scale: HillClimberWithRestarts(
            Op0Shuffle(pwr), Op1Swap2(), 2 ** i)))  # hill clim. with restarts
    ALGORITHMS.append(cast(
        Callable[[Instance, Permutations], Algorithm],
        lambda inst, pwr, i=scale: HillClimberWithRestarts(
            Op0Shuffle(pwr), Op1SwapN(), 2 ** i)))  # hill clim. with restarts
for log2_mu in range(0, 14):
    mu: int = 2 ** log2_mu
    for lambda_ in sorted({1, max(1, log2_mu), round(mu ** 0.5), mu, mu + mu,
                           4 * mu, 8 * mu}):
        if lambda_ > 65536:
            continue
        ALGORITHMS.append(cast(
            Callable[[Instance, Permutations], Algorithm],
            lambda inst, pwr, mm=mu, ll=lambda_: EA(
                Op0Shuffle(pwr), Op1Swap2(), None, mm, ll, 0.0)))
        if (mu == lambda_) and (mu in {2, 4, 32, 256, 4096}):
            for br_exp in range(1, 11):
                ALGORITHMS.append(cast(
                    Callable[[Instance, Permutations], Algorithm],
                    lambda inst, pwr, ml=mu, br=(2 ** -br_exp): EA(
                        Op0Shuffle(pwr), Op1Swap2(),
                        Op2GeneralizedAlternatingPosition(pwr), ml, ml, br)))
            for br_exp in range(2, 11):
                ALGORITHMS.append(cast(
                    Callable[[Instance, Permutations], Algorithm],
                    lambda inst, pwr, ml=mu, br=1.0 - (2 ** -br_exp): EA(
                        Op0Shuffle(pwr), Op1Swap2(),
                        Op2GeneralizedAlternatingPosition(pwr), ml, ml, br)))

for mu_lambda in [4, 32]:
    for ts in [2, 4]:
        ALGORITHMS.append(cast(
            Callable[[Instance, Permutations], Algorithm],
            lambda inst, pwr, ml=mu_lambda, sze=ts: GeneralEA(
                Op0Shuffle(pwr), Op1Swap2(),
                Op2GeneralizedAlternatingPosition(pwr), ml, ml, 2 ** -3,
                survival=TournamentWithReplacement(sze))))
        ALGORITHMS.append(cast(
            Callable[[Instance, Permutations], Algorithm],
            lambda inst, pwr, ml=mu_lambda, sze=ts: GeneralEA(
                Op0Shuffle(pwr), Op1Swap2(),
                Op2GeneralizedAlternatingPosition(pwr), ml, ml, 2 ** -3,
                survival=TournamentWithoutReplacement(sze))))
    ALGORITHMS.append(cast(
        Callable[[Instance, Permutations], Algorithm],
        lambda inst, pwr, ml=mu_lambda: GeneralEA(
            Op0Shuffle(pwr), Op1Swap2(),
            Op2GeneralizedAlternatingPosition(pwr), ml, ml, 2 ** -3,
            mating=TournamentWithoutReplacement(2))))
    ALGORITHMS.append(cast(
        Callable[[Instance, Permutations], Algorithm],
        lambda inst, pwr, ml=mu_lambda: GeneralEA(
            Op0Shuffle(pwr), Op1Swap2(),
            Op2GeneralizedAlternatingPosition(pwr), ml, ml, 2 ** -3,
            fitness=Direct(), survival=FitnessProportionateSUS())))
    ALGORITHMS.append(cast(
        Callable[[Instance, Permutations], Algorithm],
        lambda inst, pwr, ml=mu_lambda: GeneralEA(
            Op0Shuffle(pwr), Op1Swap2(),
            Op2GeneralizedAlternatingPosition(pwr), ml, ml, 2 ** -3,
            fitness=Direct(), survival=FitnessProportionateSUS(),
            mating=TournamentWithoutReplacement(2))))
    ALGORITHMS.append(cast(
        Callable[[Instance, Permutations], Algorithm],
        lambda inst, pwr, ml=mu_lambda: GeneralEA(
            Op0Shuffle(pwr), Op1Swap2(),
            Op2GeneralizedAlternatingPosition(pwr), ml, ml, 2 ** -3,
            fitness=Rank(), survival=FitnessProportionateSUS(),
            mating=TournamentWithoutReplacement(2))))

for t0 in [2.0, 4.0, 8.0, 13.0, 16.0, 32.0, 44.0, 64.0, 128.0, 148.0, 256.0]:
    for epsilon in [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]:
        ALGORITHMS.append(cast(
            Callable[[Instance, Permutations], Algorithm],
            lambda inst, pwr, t=t0, e=epsilon: SimulatedAnnealing(
                Op0Shuffle(pwr), Op1Swap2(),
                ExponentialSchedule(t, e))))

for mu_lambda in [2, 8, 32]:
    for ls_steps in [2 ** i for i in range(0, 24)]:
        ALGORITHMS.append(cast(
            Callable[[Instance, Permutations], Algorithm],
            lambda inst, pwr, ml=mu_lambda, lss=ls_steps:
                MARLS(Op0Shuffle(pwr), Op1Swap2(),
                      Op2GeneralizedAlternatingPosition(pwr),
                      ml, ml, lss)))
        ALGORITHMS.append(cast(
            Callable[[Instance, Permutations], Algorithm],
            lambda inst, pwr, ml=mu_lambda, lss=ls_steps:
                MA(Op0Shuffle(pwr),
                   Op2GeneralizedAlternatingPosition(pwr),
                   SimulatedAnnealing(
                       Op0Forward(), Op1Swap2(),
                       ExponentialSchedule(16.0, max(1e-7, round(
                           1e7 * (1 - ((0.22 / 16.0) ** (
                               1.0 / (0.8 * lss))))) / 1e7))),
                   ml, ml, lss)))

for m in [1, 2, 4, 8, 16, 32]:
    for nmax in [1, 2, 4, 8, 16, 32]:
        for max_step in [1.0, 0.5, 0.3, 0.0]:
            ALGORITHMS.append(cast(
                Callable[[Instance, Permutations], Algorithm],
                lambda inst, pwr, mm=m, ll=nmax, ms=max_step: PPA(
                    Op0Shuffle(pwr), Op1SwapTryN(pwr), mm, ll, ms)))


def run_experiment(base_dir: str = pp.join(".", "results"),
                   algorithms: Iterable[
                       Callable[[Instance, Permutations],
                                Algorithm]] = tuple(ALGORITHMS),
                   instances: Iterable[str] = INSTANCES,
                   n_runs: int = EXPERIMENT_RUNS,
                   max_time: int | None = EXPERIMENT_RUNTIME_MS,
                   max_fes: int | None = None,
                   n_threads: int | None = None) -> None:
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
    check_int_range(n_runs, "n_runs", 1, 10_000_000)
    if max_time is not None:
        check_int_range(max_time, "max_time", 1, 100_000_000_000)
    if max_fes is not None:
        check_int_range(max_fes, "max_fes", 1, 1_000_000_000_000)
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

            val: Execution | Algorithm = algor(inst, pwr)
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

    ikwargs: dict[str, Any] = {"base_dir": base_dir,
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
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = argparser(
        __file__, "Execute the JSSP example experiment.",
        "Execute an example experiment on the Job "
        "Shop Scheduling Problem (JSSP).")
    parser.add_argument(
        "dest", help="the directory where the results should be stored",
        type=Path.path, default="./results", nargs="?")
    parser.add_argument(
        "threads", help="the number of threads to use for the experiment",
        type=int, default=ex.Parallelism.ACCURATE_TIME_MEASUREMENTS,
        nargs="?")
    args: Final[argparse.Namespace] = parser.parse_args()
    run_experiment(base_dir=args.dest, n_threads=args.threads)
