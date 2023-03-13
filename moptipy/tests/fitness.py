"""Test fitness assignment processes."""


from math import inf, isfinite
from typing import Final, cast

from numpy.random import Generator, default_rng

from moptipy.algorithms.so.fitness import Fitness, FRecord, check_fitness
from moptipy.api.objective import Objective
from moptipy.api.operators import Op0
from moptipy.api.space import Space
from moptipy.tests.component import validate_component
from moptipy.utils.nputils import rand_seed_generate
from moptipy.utils.types import type_error


class _FRecord(FRecord):
    """The internal F-record."""

    def __init__(self, x, z):
        """Initialize."""
        super().__init__(x, inf)
        #: the internal id
        self.z: Final[int] = z


def validate_fitness(fitness: Fitness, objective: Objective, space: Space,
                     op0: Op0) -> None:
    """
    Validate a fitness assignment process on a given problem.

    :param fitness: the fitness assignment process
    :param objective: the objective function
    :param space: the space of solutions
    :param op0: the nullary operator
    """
    if not isinstance(fitness, Fitness):
        raise type_error(fitness, "fitness", Fitness)
    check_fitness(fitness)
    validate_component(fitness)

    random: Final[Generator] = default_rng()
    pop1: Final[list[_FRecord]] = []
    pop2: Final[list[_FRecord]] = []
    pop3: Final[list[_FRecord]] = []
    ps: Final[int] = int(1 + random.integers(48))
    for i in range(ps):
        fr: _FRecord = _FRecord(space.create(), i)
        op0.op0(random, fr.x)
        fr.f = objective.evaluate(fr.x)
        fr.it = int(random.integers(1, 20))
        if fr.fitness != inf:
            raise ValueError(f"v = {fr.fitness}, should be inf")
        pop1.append(fr)
        fr2: _FRecord = _FRecord(fr.x, fr.z)
        fr2.f = fr.f
        fr2.it = fr.it
        pop2.append(fr2)
        fr2 = _FRecord(fr.x, fr.z)
        fr2.f = fr.f
        fr2.it = fr.it
        pop3.append(fr2)

    for k in range(6):
        if k >= 3:
            # make all records identical
            fr0 = pop1[0]
            for i in range(ps):
                fr = _FRecord(fr0.x, i)
                fr.f = fr0.f
                fr.it = fr0.it
                fr.fitness = fr0.fitness
                pop1[i] = fr
                fr = _FRecord(fr0.x, i)
                fr.f = fr0.f
                fr.it = fr0.it
                fr.fitness = fr0.fitness
                pop2[i] = fr
                fr = _FRecord(fr0.x, i)
                fr.f = fr0.f
                fr.it = fr0.it
                fr.fitness = fr0.fitness
                pop3[i] = fr
        if k in (2, 4):
            for fr in pop1:
                if random.integers(2) <= 0:
                    fr.fitness = inf if random.integers(2) <= 0 else -inf

        seed: int = rand_seed_generate()
        fitness.initialize()
        fitness.assign_fitness(cast(list[FRecord], pop1), default_rng(seed))
        fitness.initialize()
        fitness.assign_fitness(cast(list[FRecord], pop3), default_rng(seed))
        pop1.sort(key=lambda r: r.z)
        pop3.sort(key=lambda r: r.z)

        for i, fr in enumerate(pop1):
            if not isinstance(fr, _FRecord):
                raise type_error(fr, f"pop[{i}]", _FRecord)
            fr2 = pop2[i]
            if fr.x is not fr2.x:
                raise ValueError("fitness assignment changed x reference!")
            if fr.it is not fr2.it:
                raise ValueError(
                    f"fitness assignment assigned rec.it to {fr.it}!")
            if fr.f is not fr2.f:
                raise ValueError(
                    f"fitness assignment assigned rec.f to {fr.f}!")
            if not isinstance(fr.fitness, int | float):
                raise type_error(fr.fitness, "rec.fitness", (int, float))
            if not isfinite(fr.fitness):
                raise ValueError(
                    f"rec.fitness should be finite, but is {fr.fitness}")
            fr2 = pop3[i]
            if (fr2.fitness != fr.fitness) or (fr2.f is not fr.f) or \
                    (fr2.it is not fr.it) or (fr2.x is not fr.x):
                raise ValueError(f"inconsistency detected when repeating "
                                 f"fitness assignment: {str(fr2)!r} != "
                                 f"{str(fr)!r} at index {i} of population "
                                 f"of length {len(pop1)}.")
