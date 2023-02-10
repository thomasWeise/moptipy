"""Compute the end result of one mock run."""

from dataclasses import dataclass
from math import ceil, inf
from typing import Final

from numpy.random import Generator

from moptipy.evaluation.end_results import EndResult
from moptipy.mock.components import (
    Algorithm,
    BasePerformance,
    Experiment,
    Instance,
)
from moptipy.utils.console import logger
from moptipy.utils.nputils import rand_generator
from moptipy.utils.types import check_int_range, type_error


def end_result(performance: BasePerformance, seed: int,
               max_fes: int | None = None,
               max_time_millis: int | None = None) -> EndResult:
    """
    Compute the end result of a mock run.

    :param performance: the performance record
    :param seed: the random seed
    :param max_fes: the maximum number of FEs
    :param max_time_millis: the maximum time
    :returns: the end result record
    """
    if not isinstance(performance, BasePerformance):
        raise type_error(performance, "performance", BasePerformance)

    limit_time: int | float = inf
    limit_fes: int | float = inf
    if max_time_millis is not None:
        limit_time = check_int_range(
            max_time_millis, "max_time_millis", 11, 1_000_000_000_000_000)
    if max_fes is not None:
        limit_fes = check_int_range(
            max_fes, "max_fes", 11, 1_000_000_000_000_000)

    # The random number generator is determined by the seed.
    random: Final[Generator] = rand_generator(seed)

    # The speed also has some slight jitter.
    jitter: Final[float] = performance.jitter
    speed: float = -1
    while (speed <= 0) or (speed >= 1):
        speed = random.normal(loc=performance.speed, scale=0.01 * jitter)

    # total_time ~ total_fes * (performance.speed ** 3)
    total_time: int
    total_fes: int
    trials: int
    if max_time_millis:
        total_time = int(max_time_millis + abs(random.normal(
            loc=0, scale=5 * jitter)))
        total_fes = -1
        trials = 0
        while ((total_fes <= 100) or (total_fes > limit_fes)) \
                and (trials < 10000):
            trials += 1
            total_fes = int(random.normal(
                loc=max(10.0, total_time / (speed ** 3)),
                scale=max(100.0, 200.0 / speed)))
        if trials >= 10000:
            total_fes = int(min(limit_fes, 10000.0))
    else:
        total_fes = max_fes if max_fes else 1_000_000
        total_time = -1
        trials = 0
        while ((total_time <= 10) or (total_time > limit_time)) \
                and (trials < 10000):
            total_time = int(random.normal(
                loc=max(10.0, total_fes * (speed ** 3)),
                scale=max(10.0, 100.0 / speed)))
        if trials >= 10000:
            total_time = int(min(limit_time, 10000.0))

    # We now look for the vicinity of the local optimum that will be found.
    # We use the quality to determine which attractor to use.
    # Then we will sample a solution between the next lower and next higher
    # attractor, again using the jitter and quality.

    # First, add some jitter to the quality.
    qual: float = -1
    while (qual <= 0) or (qual >= 1):
        qual = random.normal(loc=performance.performance, scale=0.02 * jitter)

    # Second, find the right attractor and remember it in base.
    att: Final[tuple[int, ...]] = performance.instance.attractors
    attn: Final[int] = len(att)
    att_index: int = -1
    best: Final[int] = performance.instance.best
    worst: Final[int] = performance.instance.worst
    while (att_index < 0) or (att_index >= (attn - 1)):
        att_index = int(random.normal(loc=attn * (qual ** 1.7),
                                      scale=jitter ** 0.9))
    base: Final[int] = att[att_index]

    # Third, choose the ends of the intervals in which we can jitter.
    jit_end: int = min(int(base + 0.6 * (att[att_index + 1] - base)), worst)
    jit_start: int = base
    if att_index > 0:
        jit_start = int(0.5 + ceil(base - 0.6 * (base - att[att_index - 1])))
    jit_start = max(jit_start, best)

    # Now determine best_f.
    best_f: int = -1
    while (best_f < jit_start) or (best_f > jit_end) \
            or (best_f < best) or (best_f > worst):
        uni: float = -1
        while (uni <= 0) or (uni >= 1):
            uni = abs(random.normal(loc=0, scale=jitter))
        best_f = int(round(base - uni * (base - jit_start))) \
            if random.uniform(low=0, high=1) < qual else \
            int(round(base + uni * (jit_end - base)))

    # Finally, we need to compute the time we have used.
    fact: float = -1
    while (fact <= 0) or (fact >= 1):
        fact = 1 - random.exponential(scale=(att_index + 1) / (attn + 1))

    last_improvement_fe: int = -1
    while (last_improvement_fe <= 0) or (last_improvement_fe >= total_fes):
        last_improvement_fe = int(random.normal(
            loc=total_fes * fact, scale=total_fes * 0.05 * jitter))

    last_improvement_time: int = -1
    while (last_improvement_time <= 0) \
            or (last_improvement_time >= total_time):
        last_improvement_time = int(random.normal(
            loc=total_time * fact, scale=total_time * 0.05 * jitter))

    res: Final[EndResult] = EndResult(
        algorithm=performance.algorithm.name,
        instance=performance.instance.name,
        rand_seed=seed,
        best_f=best_f,
        last_improvement_fe=last_improvement_fe,
        last_improvement_time_millis=last_improvement_time,
        total_fes=total_fes,
        total_time_millis=total_time,
        goal_f=performance.instance.best,
        max_fes=max_fes,
        max_time_millis=max_time_millis)
    return res


@dataclass(frozen=True, init=False, order=True)
class EndResults:
    """An immutable set of end results."""

    #: The experiment.
    experiment: Experiment
    #: The end results.
    results: tuple[EndResult, ...]
    #: The maximum permitted FEs.
    max_fes: int | None
    #: The maximum permitted milliseconds.
    max_time_millis: int | None
    #: the results per algorithm
    __results_for_algo: dict[str | Algorithm, tuple[EndResult, ...]]
    #: the results per instance
    __results_for_inst: dict[str | Instance, tuple[EndResult, ...]]

    def __init__(self,
                 experiment: Experiment,
                 results: tuple[EndResult, ...],
                 max_fes: int | None = None,
                 max_time_millis: int | None = None):
        """
        Create a mock results of an experiment.

        :param experiment: the experiment
        :param results: the end results
        :param max_fes: the maximum permitted FEs
        :param max_time_millis: the maximum permitted milliseconds.
        """
        if not isinstance(experiment, Experiment):
            raise type_error(experiment, "experiment", Experiment)
        object.__setattr__(self, "experiment", experiment)

        per_algo: Final[dict[str | Algorithm, list[EndResult]]] = {}
        per_inst: Final[dict[str | Instance, list[EndResult]]] = {}
        if not isinstance(results, tuple):
            raise type_error(results, "results", tuple)
        if len(results) <= 0:
            raise ValueError("end_results must not be empty.")
        for a in results:
            if not isinstance(a, EndResult):
                raise type_error(a, "element of results", EndResult)
            aa = experiment.get_algorithm(a.algorithm)
            if aa in per_algo:
                per_algo[aa].append(a)
            else:
                per_algo[aa] = [a]
            ii = experiment.get_instance(a.instance)
            if ii in per_inst:
                per_inst[ii].append(a)
            else:
                per_inst[ii] = [a]

        object.__setattr__(self, "results", results)

        pa: dict[str | Algorithm, tuple[EndResult, ...]] = {}
        for ax in experiment.algorithms:
            lax: list[EndResult] = per_algo[ax]
            lax.sort()
            pa[ax.name] = pa[ax] = tuple(lax)
        pi: dict[str | Instance, tuple[EndResult, ...]] = {}
        for ix in experiment.instances:
            lix: list[EndResult] = per_inst[ix]
            lix.sort()
            pi[ix.name] = pi[ix] = tuple(lix)

        object.__setattr__(self, "_EndResults__results_for_algo", pa)
        object.__setattr__(self, "_EndResults__results_for_inst", pi)

        if max_fes is not None:
            check_int_range(max_fes, "max_fes", 1, 1_000_000_000_000)
        object.__setattr__(self, "max_fes", max_fes)

        if max_time_millis is not None:
            check_int_range(
                max_time_millis, "max_time_millis", 1, 1_000_000_000_000_000)
        object.__setattr__(self, "max_time_millis", max_time_millis)

    @staticmethod
    def create(experiment: Experiment,
               max_fes: int | None = None,
               max_time_millis: int | None = None) -> "EndResults":
        """
        Create the end results for a given experiment.

        :param experiment: the experiment
        :param max_fes: the maximum number of FEs
        :param max_time_millis: the maximum time
        :returns: the end results
        """
        if not isinstance(experiment, Experiment):
            raise type_error(experiment, "experiment", Experiment)
        logger(
            "now creating all end results for an experiment with "
            f"{len(experiment.algorithms)} algorithms, "
            f"{len(experiment.instances)} instances, and "
            f"{len(experiment.per_instance_seeds[0])} runs per setup.")

        if max_fes is not None:
            check_int_range(max_fes, "max_fes", 1, 1_000_000_000_000)

        if max_time_millis is not None:
            check_int_range(
                max_time_millis, "max_time_millis", 1, 1_000_000_000_000_000)
        results: list[EndResult] = []
        for per in experiment.applications:
            for seed in experiment.seeds_for_instance(per.instance):
                results.append(end_result(performance=per,
                                          seed=seed,
                                          max_fes=max_fes,
                                          max_time_millis=max_time_millis))
        results.sort()

        res: Final[EndResults] = EndResults(experiment=experiment,
                                            results=tuple(results),
                                            max_fes=max_fes,
                                            max_time_millis=max_time_millis)
        logger(f"finished creating all {len(res.results)} end results.")
        return res

    def results_for_algorithm(self, algorithm: str | Algorithm) \
            -> tuple[EndResult, ...]:
        """
        Get the end results per algorithm.

        :param algorithm: the algorithm
        :returns: the end results
        """
        return self.__results_for_algo[algorithm]

    def results_for_instance(self, instance: str | Instance) \
            -> tuple[EndResult, ...]:
        """
        Get the end results per instance.

        :param instance: the instance
        :returns: the end results
        """
        return self.__results_for_inst[instance]
