"""Generate random mock experiment parameters."""

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any, Final, Iterable

from numpy.random import Generator

from moptipy.utils.console import logger
from moptipy.utils.nputils import rand_generator, rand_seeds_from_str
from moptipy.utils.strings import sanitize_name
from moptipy.utils.types import check_int_range, type_error


def fixed_random_generator() -> Generator:
    """
    Get the single, fixed random generator for the dummy experiment API.

    :returns: the random number generator
    """
    if not hasattr(fixed_random_generator, "gen"):
        setattr(fixed_random_generator, "gen", rand_generator(1))
    return getattr(fixed_random_generator, "gen")


def _random_name(namelen: int,
                 random: Generator = fixed_random_generator()) -> str:
    """
    Generate a random name of a given length.

    :param namelen: the length of the name
    :param random: a random number generator
    :returns: a name of the length
    """
    check_int_range(namelen, "namelen", 1, 1_000)
    namer: Final[tuple[str, str, str]] = ("bcdfghjklmnpqrstvwxyz", "aeiou",
                                          "0123456789")
    name = ["x"] * namelen
    index: int = 0
    n_done: bool = False
    for i in range(namelen):
        namee = namer[index]
        name[i] = namee[random.integers(len(namee))]
        n_done = n_done or (i == 2)
        if index <= 0:
            index += 1
        else:
            if (index == 2) and (random.integers(2) <= 0):
                continue
            index = int((index + 1 + random.integers(2)) % len(namer))
            if n_done:
                while index == 2:
                    index = 1 - min(1, random.integers(6))

    return "".join(name)


def __append_not_allowed(forbidden,
                         dest: set[str | float | int]) -> None:
    """
    Append items to the set of not-allowed values.

    :param forbidden: the forbidden elements
    :param dest: the set to append to
    """
    if not forbidden:
        return
    if isinstance(forbidden, str | int | float):
        dest.add(forbidden)
    elif isinstance(forbidden, Instance):
        dest.add(forbidden.name)
        dest.add(forbidden.hardness)
        dest.add(forbidden.jitter)
        dest.add(forbidden.scale)
        dest.add(forbidden.best)
        dest.add(forbidden.worst)
    elif isinstance(forbidden, Algorithm):
        dest.add(forbidden.name)
        dest.add(forbidden.strength)
        dest.add(forbidden.jitter)
    elif isinstance(forbidden, Iterable):
        for item in forbidden:
            __append_not_allowed(item, dest)
    else:
        raise type_error(forbidden, "element to add",
                         (str, int, float, Instance, Algorithm, Iterable))


def _make_not_allowed(forbidden: Iterable[str | float | int | Iterable[
        Any]] | None = None) -> set[str | float | int]:
    """
    Create a set of not-allowed values.

    :param forbidden: the forbidden elements
    :returns: the set of not-allowed values
    """
    not_allowed: set[Any] = set()
    __append_not_allowed(forbidden, not_allowed)
    return not_allowed


@dataclass(frozen=True, init=False, order=True)
class Instance:
    """An immutable instance description record."""

    #: The instance name.
    name: str
    #: The instance hardness, in (0, 1), larger values are worst
    hardness: float
    #: The instance jitter, in (0, 1), larger values are worst
    jitter: float
    #: The instance scale, in (0, 1), larger values are worst
    scale: float
    #: The best (smallest) possible objective value
    best: int
    #: The worst (largest) possible objective value
    worst: int
    #: The set of attractors, i.e., local optima - including best and worst
    attractors: tuple[int, ...]

    def __init__(self,
                 name: str,
                 hardness: float,
                 jitter: float,
                 scale: float,
                 best: int,
                 worst: int,
                 attractors: tuple[int, ...]):
        """
        Create a mock problem instance description.

        :param name: the instance name
        :param hardness: the instance hardness
        :param jitter: the instance jitter
        :param scale: the instance scale
        :param best: the best (smallest) possible objective value
        :param worst: the worst (largest) possible objective value
        :param attractors: the set of attractors, i.e., local
            optima - including best and worst
        """
        if not isinstance(name, str):
            raise type_error(name, "name", str)
        if name != sanitize_name(name):
            raise ValueError(f"Invalid name {name!r}.")
        object.__setattr__(self, "name", name)

        if not isinstance(hardness, float):
            raise type_error(hardness, "hardness", float)
        if (not isfinite(hardness)) or (hardness <= 0) or (hardness >= 1):
            raise ValueError(
                f"hardness must be in (0, 1), but is {hardness}.")
        object.__setattr__(self, "hardness", hardness)

        if not isinstance(jitter, float):
            raise type_error(jitter, "jitter", float)
        if (not isfinite(jitter)) or (jitter <= 0) or (jitter >= 1):
            raise ValueError(
                f"jitter must be in (0, 1), but is {jitter}.")
        object.__setattr__(self, "jitter", jitter)

        if not isinstance(scale, float):
            raise type_error(scale, "scale", float)
        if (not isfinite(scale)) or (scale <= 0) or (scale >= 1):
            raise ValueError(
                f"scale must be in (0, 1), but is {scale}.")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "best", check_int_range(
            best, "best", 1, 1_000_000_000))
        object.__setattr__(self, "worst", check_int_range(
            worst, "worst", best + 8, 1_000_000_000))

        if not isinstance(attractors, tuple):
            raise type_error(attractors, "attractors", tuple)
        if len(attractors) < 4:
            raise ValueError("attractors must contain at least 2 values,"
                             f" but contains only {len(attractors)}.")
        if attractors[0] != best:
            raise ValueError(
                f"attractors[0] must be {best}, but is {attractors[0]}")
        if attractors[-1] != worst:
            raise ValueError(
                f"attractors[-1] must be {worst}, but is {attractors[-1]}")
        prev = -1
        for att in attractors:
            check_int_range(att, "each attractor", max(best, prev + 1), worst)
            prev = att
        object.__setattr__(self, "attractors", attractors)

    @staticmethod
    def create(n: int,
               forbidden: Any | None = None,
               random: Generator = fixed_random_generator()) \
            -> tuple["Instance", ...]:
        """
        Create a set of fixed problem instances.

        :param n: the number of instances to generate
        :param random: a random number generator
        :param forbidden: the forbidden names and hardnesses
        :returns: a tuple of instances
        """
        check_int_range(n, "n", 1, 1_000_000)

        not_allowed: Final[set[str | float]] = \
            _make_not_allowed(forbidden)
        names: list[str] = []
        hardnesses: list[float] = []
        jitters: list[float] = []
        scales: list[float] = []

        # First we choose a unique name.
        max_name_len: int = int(max(2, ceil(n / 6)))
        trials: int = 0
        while len(names) < n:
            trials += 1
            if trials > 1000:
                trials = 1
                max_name_len += 1
            nv = _random_name(int(3 + random.integers(max_name_len)), random)
            if nv not in not_allowed:
                names.append(nv)
                not_allowed.add(nv)

        # Now we pick an instance hardness.
        limit: int = 2000
        trials = 0
        while len(hardnesses) < n:
            v: float = -1
            while (v <= 0) or (v >= 1) or (not isfinite(v)):
                trials += 1
                if trials > 1000:
                    trials = 0
                    limit *= 2
                v = (int(random.uniform(1, limit)) / limit) \
                    if random.integers(4) <= 0 else \
                    (int(random.normal(loc=0.5, scale=0.25) * limit) / limit)
            if v not in not_allowed:
                hardnesses.append(v)
                not_allowed.add(v)

        # Now we pick an instance jitter.
        limit = 2000
        trials = 0
        while len(jitters) < n:
            v = -1
            while (v <= 0) or (v >= 1) or (not isfinite(v)):
                trials += 1
                if trials > 1000:
                    trials = 0
                    limit *= 2
                v = (int(random.uniform(1, limit)) / limit) \
                    if random.integers(4) <= 0 else \
                    (int(random.normal(loc=0.5, scale=0.2) * limit) / limit)
            if v not in not_allowed:
                jitters.append(v)
                not_allowed.add(v)

        # Now we choose a scale.
        limit = 2000
        trials = 0
        while len(scales) < n:
            v = -1
            while (v <= 0) or (v >= 1) or (not isfinite(v)):
                trials += 1
                if trials > 1000:
                    trials = 0
                    limit *= 2
                v = (int(random.uniform(1, limit)) / limit) \
                    if random.integers(4) <= 0 else \
                    (int(random.normal(loc=0.5, scale=0.2) * limit) / limit)
            if v not in not_allowed:
                scales.append(v)
                not_allowed.add(v)

        # We choose the global optimum and the worst objective value.
        trials = 0
        scale: int = 1000
        loc: int = 1000
        limits: list[int] = []
        while len(limits) < (2 * n):
            tbound: int = -1
            while (tbound <= 0) or (tbound >= 1_000_000_000):
                trials += 1
                if trials > 1000:
                    trials = 0
                    scale += (scale // 3)
                    loc *= 2
                tbound = int(random.normal(
                    loc=random.integers(low=1, high=4) * loc, scale=scale))
            permitted: bool = True
            for b in limits:
                if abs(b - tbound) <= 41:
                    permitted = False
                    break
            if permitted and (tbound not in not_allowed):
                limits.append(tbound)
                not_allowed.add(tbound)

        result: list[Instance] = []
        attdone: set[int] = set()

        for i in range(n, 0, -1):
            trials = 0
            b1 = b2 = -1
            while trials < 10:
                trials += 1
                b1 = limits.pop(random.integers(i * 2))
                b2 = limits.pop(random.integers(i * 2 - 1))
                if b1 > b2:
                    b1, b2 = b2, b1
                if i <= 1:
                    break
                if (b1 * max(2.0, 0.4 * (10 - trials))) < b2:
                    break
                limits.append(b1)
                limits.append(b2)
            attdone.clear()
            attdone.add(b1)
            attdone.add(b2)

            # Now we make sure that there are at least 6 attractors.
            trials = 0
            min_dist = max(7.0, 0.07 * (b2 - b1))
            while ((len(attdone) < 6) or (random.integers(7) > 0)) \
                    and (trials < 20000):
                a = -1
                while ((a <= b1) or (a >= b2) or (a in attdone)) \
                        and (trials < 20000):
                    trials += 1
                    a = int(random.integers(low=b1 + 1, high=b2 - 1))
                if (a <= b1) or (a >= b2):
                    continue
                ok = True
                for aa in attdone:
                    if abs(aa - a) < min_dist:
                        ok = False
                        break
                if ok:
                    attdone.add(a)
                elif (trials % 1000) <= 0:
                    min_dist = max(5, 0.5 * min_dist)

            result.append(Instance(
                name=names.pop(random.integers(i)),
                hardness=hardnesses.pop(random.integers(i)),
                jitter=jitters.pop(random.integers(i)),
                scale=scales.pop(random.integers(i)),
                best=b1,
                worst=b2,
                attractors=tuple(sorted(attdone))))
        result.sort()
        logger(f"finished creating {n} instances.")
        return tuple(result)


@dataclass(frozen=True, init=False, order=True)
class Algorithm:
    """An immutable algorithm description record."""

    #: The algorithm name.
    name: str
    #: The algorithm strength, in (0, 1), larger values are worst
    strength: float
    #: The algorithm jitter, in (0, 1), larger values are worst
    jitter: float
    #: The algorithm complexity, in (0, 1), larger values are worst
    complexity: float

    def __init__(self,
                 name: str,
                 strength: float,
                 jitter: float,
                 complexity: float):
        """
        Create a mock algorithm description record.

        :param name: the algorithm name
        :param strength: the algorithm strength
        :param jitter: the algorithm jitter
        :param complexity: the algorithm complexity
        """
        if not isinstance(name, str):
            raise type_error(name, "name", str)
        if name != sanitize_name(name):
            raise ValueError(f"Invalid name {name!r}.")
        object.__setattr__(self, "name", name)

        if not isinstance(strength, float):
            raise type_error(strength, "strength", float)
        if (not isfinite(strength)) or (strength <= 0) or (strength >= 1):
            raise ValueError(
                f"strength must be in (0, 1), but is {strength}.")
        object.__setattr__(self, "strength", strength)

        if not isinstance(jitter, float):
            raise type_error(jitter, "jitter", float)
        if (not isfinite(jitter)) or (jitter <= 0) or (jitter >= 1):
            raise ValueError(
                f"jitter must be in (0, 1), but is {jitter}.")
        object.__setattr__(self, "jitter", jitter)

        if not isinstance(complexity, float):
            raise type_error(complexity, "complexity", float)
        if (not isfinite(complexity)) or (complexity <= 0) \
                or (complexity >= 1):
            raise ValueError(
                f"complexity must be in (0, 1), but is {complexity}.")
        object.__setattr__(self, "complexity", complexity)

    @staticmethod
    def create(n: int,
               forbidden: Any | None = None,
               random: Generator = fixed_random_generator()) \
            -> tuple["Algorithm", ...]:
        """
        Create a set of fixed mock algorithms.

        :param n: the number of algorithms to generate
        :param random: a random number generator
        :param forbidden: the forbidden names and strengths and so on
        :returns: a tuple of algorithms
        """
        check_int_range(n, "n", 1)
        logger(f"now creating {n} algorithms.")

        not_allowed: Final[set[str | float]] = \
            _make_not_allowed(forbidden)
        names: list[str] = []
        strengths: list[float] = []
        jitters: list[float] = []
        complexities: list[float] = []

        prefixes: Final[tuple[str, ...]] = ("aco", "bobyqa", "cma-es", "de",
                                            "ea", "eda", "ga", "gp", "hc",
                                            "ma", "pso", "rs", "rw", "sa",
                                            "umda")
        suffixes: Final[tuple[str, ...]] = ("1swap", "2swap", "µ")

        max_name_len: int = int(max(2, ceil(n / 6)))
        trials: int = 0
        while len(names) < n:
            trials += 1
            if trials > 1000:
                trials = 1
                max_name_len += 1

            name_mode = random.integers(5)
            if name_mode < 2:
                nva = _random_name(int(3 + random.integers(max_name_len)),
                                   random)
                if nva in not_allowed:
                    continue
                if name_mode == 1:
                    nvb = suffixes[random.integers(len(suffixes))]
                    nv = f"{nva}_{nvb}"
                else:
                    nv = nva
                if nv in not_allowed:
                    continue
                not_allowed.add(nva)
                not_allowed.add(nv)
                names.append(nv)
                continue

            nva = prefixes[random.integers(len(prefixes))]
            if name_mode == 3:
                nvb = _random_name(int(3 + random.integers(
                    max_name_len)), random)
                if nvb in not_allowed:
                    continue
                nv = f"{nva}_{nvb}"
            elif name_mode == 4:
                nvb = suffixes[random.integers(len(suffixes))]
                nv = f"{nva}_{nvb}"
            else:
                nv = nva
                nvb = ""

            if nv in not_allowed:
                continue
            names.append(nv)
            not_allowed.add(nv)
            if name_mode == 3:
                not_allowed.add(nvb)

        limit: int = 2000
        trials = 0
        while len(strengths) < n:
            v: float = -1
            while (v <= 0) or (v >= 1) or (not isfinite(v)):
                trials += 1
                if trials > 1000:
                    trials = 0
                    limit *= 2
                v = (int(random.uniform(1, limit)) / limit) \
                    if random.integers(4) <= 0 else \
                    (int(random.normal(loc=0.5, scale=0.25) * limit) / limit)
            if v not in not_allowed:
                strengths.append(v)
                not_allowed.add(v)

        limit = 2000
        trials = 0
        while len(jitters) < n:
            v = -1
            while (v <= 0) or (v >= 1) or (not isfinite(v)):
                trials += 1
                if trials > 1000:
                    trials = 0
                    limit *= 2
                v = (int(random.uniform(1, limit)) / limit) \
                    if random.integers(4) <= 0 else \
                    (int(random.normal(loc=0.5, scale=0.2) * limit) / limit)
            if v not in not_allowed:
                jitters.append(v)
                not_allowed.add(v)

        limit = 2000
        trials = 0
        while len(complexities) < n:
            v = -1
            while (v <= 0) or (v >= 1) or (not isfinite(v)):
                trials += 1
                if trials > 1000:
                    trials = 0
                    limit *= 2
                v = (int(random.uniform(1, limit)) / limit) \
                    if random.integers(4) <= 0 else \
                    (int(random.normal(loc=0.5, scale=0.2) * limit) / limit)
            if v not in not_allowed:
                complexities.append(v)
                not_allowed.add(v)

        result: list[Algorithm] = []
        for i in range(n, 0, -1):
            result.append(Algorithm(
                name=names.pop(random.integers(i)),
                strength=strengths.pop(random.integers(i)),
                jitter=jitters.pop(random.integers(i)),
                complexity=complexities.pop(random.integers(i))))
        result.sort()

        logger(f"finished creating {n} algorithms.")
        return tuple(result)


@dataclass(frozen=True, init=False, order=True)
class BasePerformance:
    """An algorithm applied to a problem instance description record."""

    #: The algorithm.
    algorithm: Algorithm
    #: The problem instance
    instance: Instance
    #: The base performance, in (0, 1), larger values are worst
    performance: float
    #: The performance jitter, in (0, 1), larger values are worst
    jitter: float
    #: The time per FE, in (0, 1), larger values are worst
    speed: float

    def __init__(self,
                 algorithm: Algorithm,
                 instance: Instance,
                 performance: float,
                 jitter: float,
                 speed: float):
        """
        Create a mock algorithm-instance application description record.

        :param algorithm: the algorithm
        :param instance: the instance
        :param performance: the base performance
        :param jitter: the performance jitter
        :param speed: the time required per FE
        """
        if not isinstance(algorithm, Algorithm):
            raise type_error(algorithm, "algorithm", Algorithm)
        object.__setattr__(self, "algorithm", algorithm)
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        object.__setattr__(self, "instance", instance)

        if not isinstance(performance, float):
            raise type_error(performance, "performance", float)
        if (not isfinite(performance)) or (performance <= 0) \
                or (performance >= 1):
            raise ValueError(
                f"performance must be in (0, 1), but is {performance}.")
        object.__setattr__(self, "performance", performance)

        if not isinstance(jitter, float):
            raise type_error(jitter, "jitter", float)
        if (not isfinite(jitter)) or (jitter <= 0) or (jitter >= 1):
            raise ValueError(
                f"jitter must be in (0, 1), but is {jitter}.")
        object.__setattr__(self, "jitter", jitter)

        if not isinstance(speed, float):
            raise type_error(speed, "speed", float)
        if (not isfinite(speed)) or (speed <= 0) or (speed >= 1):
            raise ValueError(
                f"speed must be in (0, 1), but is {speed}.")
        object.__setattr__(self, "speed", speed)

    @staticmethod
    def create(instance: Instance,
               algorithm: Algorithm,
               random: Generator = fixed_random_generator()) \
            -> "BasePerformance":
        """
        Compute the basic performance of an algorithm on a problem instance.

        :param instance: the instance tuple
        :param algorithm: the algorithm tuple
        :param random: the random number generator
        :returns: a tuple of the performance in (0, 1); bigger values are
            worse, and a jitter in (0, 1), where bigger values are worse
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        if not isinstance(algorithm, Algorithm):
            raise type_error(algorithm, "algorithm", Algorithm)
        logger("now creating base performance for algorithm "
               f"{algorithm.name} on instance {instance.name}.")

        perf: float = -1
        granularity: Final[int] = 2000
        while (perf <= 0) or (perf >= 1):
            perf = random.uniform(low=0, high=1) \
                if random.integers(20) <= 0 else \
                random.normal(
                    loc=0.5 * (instance.hardness + algorithm.strength),
                    scale=0.2 * (instance.jitter + algorithm.jitter))
            perf = int(perf * granularity) / granularity

        jit: float = -1
        while (jit <= 0) or (jit >= 1):
            jit = random.uniform(low=0, high=1) \
                if random.integers(15) <= 0 else \
                random.normal(
                    loc=0.5 * (instance.jitter + algorithm.jitter),
                    scale=0.2 * (instance.jitter + algorithm.jitter))
            jit = int(jit * granularity) / granularity

        speed: float = -1
        while (speed <= 0) or (speed >= 1):
            speed = random.uniform(low=0, high=1) \
                if random.integers(20) <= 0 else \
                random.normal(
                    loc=0.5 * (instance.scale + algorithm.complexity),
                    scale=0.2 * (instance.scale + algorithm.complexity))
            speed = int(speed * granularity) / granularity

        bp: Final[BasePerformance] = BasePerformance(algorithm=algorithm,
                                                     instance=instance,
                                                     performance=perf,
                                                     jitter=jit,
                                                     speed=speed)
        logger("finished base performance "
               f"{bp.algorithm.name}@{bp.instance.name}.")
        return bp


def get_run_seeds(instance: Instance, n_runs: int) -> tuple[int, ...]:
    """
    Get the seeds for the runs.

    :param instance: the mock instance
    :param n_runs: the number of runs
    :returns: a tuple of seeds
    """
    if not isinstance(instance, Instance):
        raise type_error(instance, "instance", Instance)
    check_int_range(n_runs, "n_runs", 1, 1_000_000)
    res: Final[tuple[int, ...]] = tuple(sorted(rand_seeds_from_str(
        string=instance.name, n_seeds=n_runs)))
    logger(f"finished creating {n_runs} seeds for instance {instance.name}.")
    return res


@dataclass(frozen=True, init=False, order=True)
class Experiment:
    """An immutable experiment description."""

    #: The instances.
    instances: tuple[Instance, ...]
    #: The algorithms.
    algorithms: tuple[Algorithm, ...]
    #: The applications of the algorithms to the instances.
    applications: tuple[BasePerformance, ...]
    #: The random seeds per instance.
    per_instance_seeds: tuple[tuple[int, ...]]
    #: the seeds per instance
    __seeds_per_inst: dict[str | Instance, tuple[int, ...]]
    #: the performance per algorithm
    __perf_per_algo: dict[str | Algorithm, tuple[BasePerformance, ...]]
    #: the performance per instance
    __perf_per_inst: dict[str | Instance, tuple[BasePerformance, ...]]
    #: the algorithm by names
    __algo_by_name: dict[str, Algorithm]
    #: the algorithm names
    algorithm_names: tuple[str, ...]
    #: the instance by names
    __inst_by_name: dict[str, Instance]
    #: the instance names
    instance_names: tuple[str, ...]

    def __init__(self,
                 instances: tuple[Instance, ...],
                 algorithms: tuple[Algorithm, ...],
                 applications: tuple[BasePerformance, ...],
                 per_instance_seeds: tuple[tuple[int, ...], ...]):
        """
        Create a mock experiment definition.

        :param algorithms: the algorithms
        :param instances: the instances
        :param applications: the applications of the algorithms to the
            instances
        :param per_instance_seeds: the seeds
        """
        if not isinstance(instances, tuple):
            raise type_error(instances, "instances", tuple)
        if len(instances) <= 0:
            raise ValueError("instances must not be empty.")
        inst_bn: dict[str, Instance] = {}
        for a in instances:
            if not isinstance(a, Instance):
                raise type_error(a, "element of instances", Instance)
            if a.name in inst_bn:
                raise ValueError(f"double instance name {a.name}.")
            inst_bn[a.name] = a
        object.__setattr__(self, "instances", instances)
        object.__setattr__(self, "_Experiment__inst_by_name", inst_bn)
        object.__setattr__(self, "instance_names",
                           tuple(sorted(inst_bn.keys())))

        if not isinstance(algorithms, tuple):
            raise type_error(algorithms, "algorithms", tuple)
        if len(algorithms) <= 0:
            raise ValueError("algorithms must not be empty.")
        algo_bn: dict[str, Algorithm] = {}
        for b in algorithms:
            if not isinstance(b, Algorithm):
                raise type_error(b, "element of algorithms", Algorithm)
            if b.name in algo_bn:
                raise ValueError(f"double algorithm name {b.name}.")
            if b.name in inst_bn:
                raise ValueError(f"instance/algorithm name {b.name} clash.")
            algo_bn[b.name] = b
        object.__setattr__(self, "algorithms", algorithms)
        object.__setattr__(self, "_Experiment__algo_by_name", algo_bn)
        object.__setattr__(self, "algorithm_names",
                           tuple(sorted(algo_bn.keys())))

        if not isinstance(applications, tuple):
            raise type_error(applications, "applications", tuple)
        if len(applications) != len(algorithms) * len(instances):
            raise ValueError(
                f"There must be {len(algorithms) * len(instances)} "
                f"applications, but found {len(applications)}.")

        perf_per_inst: dict[Instance, list[BasePerformance]] = {}
        perf_per_algo: dict[Algorithm, list[BasePerformance]] = {}

        done: set[str] = set()
        for c in applications:
            if not isinstance(c, BasePerformance):
                raise type_error(c, "element of applications", BasePerformance)
            s = c.algorithm.name + "+" + c.instance.name
            if s in done:
                raise ValueError(f"Encountered application {s} twice.")
            done.add(s)
            if c.algorithm in perf_per_algo:
                perf_per_algo[c.algorithm].append(c)
            else:
                perf_per_algo[c.algorithm] = [c]
            if c.instance in perf_per_inst:
                perf_per_inst[c.instance].append(c)
            else:
                perf_per_inst[c.instance] = [c]
        object.__setattr__(self, "applications", applications)

        pa: dict[str | Algorithm, tuple[BasePerformance, ...]] = {}
        for ax in algorithms:
            lax: list[BasePerformance] = perf_per_algo[ax]
            lax.sort()
            pa[ax.name] = pa[ax] = tuple(lax)
        pi: dict[str | Instance, tuple[BasePerformance, ...]] = {}
        for ix in instances:
            lix: list[BasePerformance] = perf_per_inst[ix]
            lix.sort()
            pi[ix.name] = pi[ix] = tuple(lix)
        object.__setattr__(self, "_Experiment__perf_per_algo", pa)
        object.__setattr__(self, "_Experiment__perf_per_inst", pi)

        if not isinstance(per_instance_seeds, tuple):
            raise type_error(per_instance_seeds, "per_instance_seeds", tuple)
        if len(per_instance_seeds) != len(instances):
            raise ValueError(
                f"There must be one entry for each of the {len(instances)} "
                "instances, but per_instance_seeds only "
                f"has {len(per_instance_seeds)}.")
        xl: int = -1
        inst_seeds: Final[dict[str | Instance, tuple[int, ...]]] = {}
        for idx, d in enumerate(per_instance_seeds):
            if not isinstance(d, tuple):
                raise type_error(d, "element of per_instance_seeds", tuple)
            if len(d) <= 0:
                raise ValueError(f"there must be at least one per "
                                 f"instance seed, but found {len(d)}.")
            if xl < 0:
                xl = len(d)
            if len(d) != xl:
                raise ValueError(f"there must be {xl} per "
                                 f"instance seeds, but found {len(d)}.")
            for e in d:
                if not isinstance(e, int):
                    raise type_error(e, "element of seeds", int)
            inst_seeds[instances[idx]] = d
            inst_seeds[instances[idx].name] = d
        object.__setattr__(self, "per_instance_seeds", per_instance_seeds)
        object.__setattr__(self, "_Experiment__seeds_per_inst", inst_seeds)

    @staticmethod
    def create(n_instances: int,
               n_algorithms: int,
               n_runs: int,
               random: Generator = fixed_random_generator()) -> "Experiment":
        """
        Create an experiment definition.

        :param n_instances: the number of instances
        :param n_algorithms: the number of algorithms
        :param n_runs: the number of per-instance runs
        :param random: the random number generator to use
        """
        check_int_range(n_instances, "n_instances", 1, 1_000_000)
        check_int_range(n_algorithms, "n_algorithms", 1, 1_000_000)
        check_int_range(n_runs, "n_runs", 1, 1_000_000)
        logger(f"now creating mock experiment with {n_algorithms} algorithms "
               f"on {n_instances} instances for {n_runs} runs.")

        insts = Instance.create(n_instances, random=random)
        algos = Algorithm.create(n_algorithms, forbidden=insts, random=random)
        app = [BasePerformance.create(i, a, random)
               for i in insts for a in algos]
        app.sort()
        seeds = [get_run_seeds(i, n_runs) for i in insts]
        res: Final[Experiment] = Experiment(instances=insts,
                                            algorithms=algos,
                                            applications=tuple(app),
                                            per_instance_seeds=tuple(seeds))
        logger(f"finished creating mock experiment with {len(res.instances)} "
               f"instances, {len(res.algorithms)} algorithms, and "
               f"{len(res.per_instance_seeds[0])} runs per instance-"
               f"algorithm combination.")
        return res

    def seeds_for_instance(self, instance: str | Instance) \
            -> tuple[int, ...]:
        """
        Get the seeds for the specified instance.

        :param instance: the instance
        :returns: the seeds
        """
        return self.__seeds_per_inst[instance]

    def instance_applications(self, instance: str | Instance) \
            -> tuple[BasePerformance, ...]:
        """
        Get the applications of the algorithms to a specific instance.

        :param instance: the instance
        :returns: the applications
        """
        return self.__perf_per_inst[instance]

    def algorithm_applications(self, algorithm: str | Algorithm) \
            -> tuple[BasePerformance, ...]:
        """
        Get the applications of an algorithm to the instances.

        :param algorithm: the algorithm
        :returns: the applications
        """
        return self.__perf_per_algo[algorithm]

    def get_algorithm(self, name: str) -> Algorithm:
        """
        Get an algorithm by name.

        :param name: the algorithm name
        :returns: the algorithm instance
        """
        return self.__algo_by_name[name]

    def get_instance(self, name: str) -> Instance:
        """
        Get an instance by name.

        :param name: the instance name
        :returns: the instance
        """
        return self.__inst_by_name[name]
