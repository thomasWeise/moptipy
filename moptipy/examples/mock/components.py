"""Generate random mock experiment parameters."""

from dataclasses import dataclass
from math import isfinite, ceil
from typing import Tuple, Final, Set, List, Optional, Iterable, Union, Any

from numpy.random import Generator

from moptipy.utils import logging
from moptipy.utils.nputils import rand_generator, rand_seeds_from_str


def fixed_random_generator() -> Generator:
    """
    Get the single, fixed random generator for the dummy experiment API.

    :returns: the random number generator
    :rtype: the generator
    """
    if not hasattr(fixed_random_generator, 'gen'):
        setattr(fixed_random_generator, "gen", rand_generator(1))
    return getattr(fixed_random_generator, 'gen')


def _random_name(namelen: int,
                 random: Generator = fixed_random_generator()) -> str:
    """
    Generate a random name of a given length.

    :param int namelen: the length of the name
    :param Generator random: a random number generator
    :returns: a name of the length
    :rtype: str
    """
    if not isinstance(namelen, int):
        raise TypeError(f"namelen must be int, but is {type(namelen)}.")
    if namelen <= 0:
        raise ValueError(f"namelen must be > 0, but is {namelen}.")
    namer: Final[Tuple[str, str, str]] = ("bcdfghjklmnpqrstvwxyz", "aeiou",
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
            if index == 2:
                if random.integers(2) <= 0:
                    continue
            index = int((index + 1 + random.integers(2)) % len(namer))
            if n_done:
                while index == 2:
                    index = 1 - min(1, random.integers(6))

    return "".join(name)


def __append_not_allowed(forbidden,
                         dest: Set[Union[str, float, int]]):
    """
    Append items to the set of not-allowed values.

    :param forbidden: the forbidden elements
    :param dest: the set to append to
    """
    if not forbidden:
        return
    if isinstance(forbidden, (str, int, float)):
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
        raise TypeError(
            f"element to add must be str, int, float, Instance, Algorithm, "
            f"or an Iterable thereof, but is {type(forbidden)}.")


def _make_not_allowed(forbidden: Optional[Iterable[Union[
    str, float, int, Iterable[Any]]]] = None) -> \
        Set[Union[str, float, int]]:
    """
    Create a set of not-allowed values.

    :param forbidden: the forbidden elements
    :returns: the set of not-allowed values
    :rtype: Set[Union[str, float]]
    """
    not_allowed: Set[Any] = set()
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
    attractors: Tuple[int, ...]

    def __init__(self,
                 name: str,
                 hardness: float,
                 jitter: float,
                 scale: float,
                 best: int,
                 worst: int,
                 attractors: Tuple[int, ...]):
        """
        Create a mock problem instance description.

        :param str name: the instance name
        :param float hardness: the instance hardness
        :param float jitter: the instance jitter
        :param float scale: the instance scale
        :param int best: the best (smallest) possible objective value
        :param int worst: the worst (largest) possible objective value
        :param Tuple[int, ...] attractors: the set of attractors, i.e., local
            optima - including best and worst
        """
        if not isinstance(name, str):
            raise TypeError(
                f"name must be str, but is {type(name)}.")
        if name != logging.sanitize_name(name):
            raise ValueError(f"Invalid name '{name}'.")
        object.__setattr__(self, "name", name)

        if not isinstance(hardness, float):
            raise TypeError(
                f"hardness must be float, but is {type(hardness)}.")
        if (not isfinite(hardness)) or (hardness <= 0) or (hardness >= 1):
            raise ValueError(
                f"hardness must be in (0, 1), but is {hardness}.")
        object.__setattr__(self, "hardness", hardness)

        if not isinstance(jitter, float):
            raise TypeError(
                f"jitter must be float, but is {type(jitter)}.")
        if (not isfinite(jitter)) or (jitter <= 0) or (jitter >= 1):
            raise ValueError(
                f"jitter must be in (0, 1), but is {jitter}.")
        object.__setattr__(self, "jitter", jitter)

        if not isinstance(scale, float):
            raise TypeError(
                f"scale must be float, but is {type(scale)}.")
        if (not isfinite(scale)) or (scale <= 0) or (scale >= 1):
            raise ValueError(
                f"scale must be in (0, 1), but is {scale}.")
        object.__setattr__(self, "scale", scale)

        if not isinstance(best, int):
            raise TypeError(
                f"best must be int, but is {type(best)}.")
        if (best <= 0) or (best >= 1_000_000_000):
            raise ValueError(
                f"best must be in 1...999999999, but is {best}.")
        object.__setattr__(self, "best", best)

        if not isinstance(worst, int):
            raise TypeError(
                f"worst must be int, but is {type(worst)}.")
        if (worst <= (best + 7)) or (worst >= 1_000_000_000):
            raise ValueError(
                f"worst must be in {best + 8}...999999999, but is {worst}.")
        object.__setattr__(self, "worst", worst)

        if not isinstance(attractors, tuple):
            raise TypeError(
                f"attractors must be Tuple, but is {type(attractors)}.")
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
            if not isinstance(att, int):
                raise TypeError(f"each attractor must be int, but "
                                f"encountered {att}, which is {type(att)}.")
            if (att < best) or (att > worst) or (att <= prev):
                raise ValueError(f"{att} not permitted after {prev} "
                                 f"for best={best} and worst={worst}.")
            prev = att
        object.__setattr__(self, "attractors", attractors)

    @staticmethod
    def create(n: int,
               forbidden: Optional[Iterable[Union[
                   str, float, Iterable[Any]]]] = None,
               random: Generator = fixed_random_generator()) \
            -> Tuple['Instance', ...]:
        """
        Create a set of fixed problem instances.

        :param int n: the number of instances to generate
        :param Generator random: a random number generator
        :param forbidden: the forbidden names and hardnesses
        :returns: a tuple of instances
        :rtype: Tuple[Instance, ...]
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be int, but is {type(n)}.")
        if n <= 0:
            raise ValueError(f"n must be > 0, but is {n}.")

        not_allowed: Final[Set[Union[str, float]]] = \
            _make_not_allowed(forbidden)
        names: List[str] = []
        hardnesses: List[float] = []
        jitters: List[float] = []
        scales: List[float] = []
        limits: List[int] = []

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

        limit: int = 2000
        trials = 0
        while len(hardnesses) < n:
            v: float = -1
            while (v <= 0) or (v >= 1) or (not isfinite(v)):
                trials += 1
                if trials > 1000:
                    trials = 0
                    limit *= 2
                if random.integers(4) <= 0:
                    v = int(random.uniform(1, limit)) / limit
                else:
                    v = int(random.normal(loc=0.5, scale=0.25) * limit) / limit
            if v not in not_allowed:
                hardnesses.append(v)
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
                if random.integers(4) <= 0:
                    v = int(random.uniform(1, limit)) / limit
                else:
                    v = int(random.normal(loc=0.5, scale=0.2) * limit) / limit
            if v not in not_allowed:
                jitters.append(v)
                not_allowed.add(v)

        limit = 2000
        trials = 0
        while len(scales) < n:
            v = -1
            while (v <= 0) or (v >= 1) or (not isfinite(v)):
                trials += 1
                if trials > 1000:
                    trials = 0
                    limit *= 2
                if random.integers(4) <= 0:
                    v = int(random.uniform(1, limit)) / limit
                else:
                    v = int(random.normal(loc=0.5, scale=0.2) * limit) / limit
            if v not in not_allowed:
                scales.append(v)
                not_allowed.add(v)

        trials = 0
        scale: int = 1000
        loc: int = 1000
        while len(limits) < (2 * n):
            tbound: int = -1
            while (tbound <= 0) or (tbound >= 1_000_000_000):
                trials += 1
                if trials > 1000:
                    trials = 0
                    scale += (scale // 3)
                    loc *= 2
                tbound = int(random.normal(loc=loc, scale=scale))
            permitted: bool = True
            for b in limits:
                if abs(b - tbound) <= 12:
                    permitted = False
                    break
            if permitted and (tbound not in not_allowed):
                limits.append(tbound)
                not_allowed.add(tbound)

        result: List[Instance] = []
        attdone: Set[int] = set()
        for i in range(n, 0, -1):
            b1 = limits.pop(random.integers(i * 2))
            b2 = limits.pop(random.integers(i * 2 - 1))

            if b1 > b2:
                b1, b2 = b2, b1
            attdone.clear()
            attdone.add(b1)
            attdone.add(b2)

            trials = 0
            while (len(attdone) < 6) or (random.integers(7) > 0) \
                    and (trials < 10000):
                a = -1
                while ((a <= b1) or (a >= b2) or (a in attdone)) \
                        and (trials < 10000):
                    trials += 1
                    a = random.integers(low=b1 + 1, high=b2 - 1)
                if (a <= b1) or (a >= b2):
                    continue
                ok = True
                for aa in attdone:
                    if abs(aa - a) < 5:
                        ok = False
                        break
                if ok:
                    attdone.add(a)

            result.append(Instance(
                name=names.pop(random.integers(i)),
                hardness=hardnesses.pop(random.integers(i)),
                jitter=jitters.pop(random.integers(i)),
                scale=scales.pop(random.integers(i)),
                best=b1,
                worst=b2,
                attractors=tuple(sorted(attdone))))
        result.sort()
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

        :param str name: the algorithm name
        :param float strength: the algorithm strength
        :param float jitter: the algorithm jitter
        :param float complexity: the algorithm complexity
        """
        if not isinstance(name, str):
            raise TypeError(
                f"name must be str, but is {type(name)}.")
        if name != logging.sanitize_name(name):
            raise ValueError(f"Invalid name '{name}'.")
        object.__setattr__(self, "name", name)

        if not isinstance(strength, float):
            raise TypeError(
                f"strength must be float, but is {type(strength)}.")
        if (not isfinite(strength)) or (strength <= 0) or (strength >= 1):
            raise ValueError(
                f"strength must be in (0, 1), but is {strength}.")
        object.__setattr__(self, "strength", strength)

        if not isinstance(jitter, float):
            raise TypeError(
                f"jitter must be float, but is {type(jitter)}.")
        if (not isfinite(jitter)) or (jitter <= 0) or (jitter >= 1):
            raise ValueError(
                f"jitter must be in (0, 1), but is {jitter}.")
        object.__setattr__(self, "jitter", jitter)

        if not isinstance(complexity, float):
            raise TypeError(
                f"complexity must be float, but is {type(complexity)}.")
        if (not isfinite(complexity)) or (complexity <= 0) \
                or (complexity >= 1):
            raise ValueError(
                f"complexity must be in (0, 1), but is {complexity}.")
        object.__setattr__(self, "complexity", complexity)

    @staticmethod
    def create(n: int,
               forbidden: Optional[Iterable[Union[
                   str, float, Iterable[Any]]]] = None,
               random: Generator = fixed_random_generator()) \
            -> Tuple['Algorithm', ...]:
        """
        Create a set of fixed mock algorithms.

        :param int n: the number of algorithms to generate
        :param Generator random: a random number generator
        :param forbidden: the forbidden names and strengths and so on
        :returns: a tuple of algorithms
        :rtype: Tuple[Algorithm, ...]
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be int, but is {type(n)}.")
        if n <= 0:
            raise ValueError(f"n must be > 0, but is {n}.")

        not_allowed: Final[Set[Union[str, float]]] = \
            _make_not_allowed(forbidden)
        names: List[str] = []
        strengths: List[float] = []
        jitters: List[float] = []
        complexities: List[float] = []

        prefixes: Final[Tuple[str, ...]] = ('aco', 'bobyqa', 'cma-es', 'de',
                                            'ea', 'eda', 'ga', 'gp', 'hc',
                                            'ma', 'pso', 'rs', 'rw', 'sa',
                                            'umda')
        suffixes: Final[Tuple[str, ...]] = ("1swap", "2swap", "µ")

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
                    if random.integers(4) <= 0:
                        v = int(random.uniform(1, limit)) / limit
                    else:
                        v = int(random.normal(loc=0.5,
                                              scale=0.25) * limit) / limit
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
                    if random.integers(4) <= 0:
                        v = int(random.uniform(1, limit)) / limit
                    else:
                        v = int(random.normal(loc=0.5,
                                              scale=0.2) * limit) / limit
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
                    if random.integers(4) <= 0:
                        v = int(random.uniform(1, limit)) / limit
                    else:
                        v = int(random.normal(loc=0.5,
                                              scale=0.2) * limit) / limit
                if v not in not_allowed:
                    complexities.append(v)
                    not_allowed.add(v)

        result: List[Algorithm] = []
        for i in range(n, 0, -1):
            result.append(Algorithm(
                name=names.pop(random.integers(i)),
                strength=strengths.pop(random.integers(i)),
                jitter=jitters.pop(random.integers(i)),
                complexity=complexities.pop(random.integers(i))))
        result.sort()
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

        :param Algorithm algorithm: the algorithm
        :param Instance instance: the instance
        :param float performance: the base performance
        :param float jitter: the performance jitter
        :param float speed: the time required per FE
        """
        if not isinstance(algorithm, Algorithm):
            raise TypeError(
                f"algorithm must be Algorithm is {type(algorithm)}.")
        object.__setattr__(self, "algorithm", algorithm)
        if not isinstance(instance, Instance):
            raise TypeError(
                f"instance must be Instance is {type(instance)}.")
        object.__setattr__(self, "instance", instance)

        if not isinstance(performance, float):
            raise TypeError(
                f"performance must be float, but is {type(performance)}.")
        if (not isfinite(performance)) or (performance <= 0) \
                or (performance >= 1):
            raise ValueError(
                f"performance must be in (0, 1), but is {performance}.")
        object.__setattr__(self, "performance", performance)

        if not isinstance(jitter, float):
            raise TypeError(
                f"jitter must be float, but is {type(jitter)}.")
        if (not isfinite(jitter)) or (jitter <= 0) or (jitter >= 1):
            raise ValueError(
                f"jitter must be in (0, 1), but is {jitter}.")
        object.__setattr__(self, "jitter", jitter)

        if not isinstance(speed, float):
            raise TypeError(
                f"speed must be float, but is {type(speed)}.")
        if (not isfinite(speed)) or (speed <= 0) or (speed >= 1):
            raise ValueError(
                f"speed must be in (0, 1), but is {speed}.")
        object.__setattr__(self, "speed", speed)

    @staticmethod
    def create(instance: Instance,
               algorithm: Algorithm,
               random: Generator = fixed_random_generator()) \
            -> 'BasePerformance':
        """
        Compute the basic performance of an algorithm on a problem instance.

        :param instance: the instance tuple
        :param algorithm: the algorithm tuple
        :param random: the random number generator
        :returns: a tuple of the performance in (0, 1); bigger values are
            worse, and a jitter in (0, 1), where bigger values are worse
        :rtype: Tuple[float, float]
        """
        if not isinstance(instance, Instance):
            raise TypeError(
                f"instance must be Instance, but is {type(instance)}.")
        if not isinstance(algorithm, Algorithm):
            raise TypeError(
                f"algorithm must be Algorithm, but is {type(algorithm)}.")
        perf: float = -1
        granularity: Final[int] = 2000
        while (perf <= 0) or (perf >= 1):
            if random.integers(20) <= 0:
                perf = random.uniform(low=0, high=1)
            else:
                perf = random.normal(
                    loc=0.5 * (instance.hardness + algorithm.strength),
                    scale=0.2 * (instance.jitter + algorithm.jitter))
            perf = int(perf * granularity) / granularity

        jit: float = -1
        while (jit <= 0) or (jit >= 1):
            if random.integers(15) <= 0:
                jit = random.uniform(low=0, high=1)
            else:
                jit = random.normal(
                    loc=0.5 * (instance.jitter + algorithm.jitter),
                    scale=0.2 * (instance.jitter + algorithm.jitter))
            jit = int(jit * granularity) / granularity

        speed: float = -1
        while (speed <= 0) or (speed >= 1):
            if random.integers(20) <= 0:
                speed = random.uniform(low=0, high=1)
            else:
                speed = random.normal(
                    loc=0.5 * (instance.scale + algorithm.complexity),
                    scale=0.2 * (instance.scale + algorithm.complexity))
            speed = int(speed * granularity) / granularity

        return BasePerformance(algorithm=algorithm,
                               instance=instance,
                               performance=perf,
                               jitter=jit,
                               speed=speed)


def get_run_seeds(instance: Instance, n_runs: int) -> Tuple[int, ...]:
    """
    Get the seeds for the runs.

    :param Instance instance: the mock instance
    :param int n_runs: the number of runs
    :returns: a tuple of seeds
    :rtype: Tuple[int, ...]
    """
    if not isinstance(instance, Instance):
        raise TypeError(
            f"instance must be Instance, but is {type(instance)}.")
    if not isinstance(n_runs, int):
        raise TypeError(f"n_runs must be int, but is {type(n_runs)}.")
    if n_runs <= 0:
        raise ValueError(f"n_runs must be > 0, but is {n_runs}.")
    return tuple(rand_seeds_from_str(string=instance.name, n_seeds=n_runs))
