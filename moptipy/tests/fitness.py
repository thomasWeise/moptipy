"""Test fitness assignment processes."""


from math import inf, isfinite
from typing import Final, List, cast

from numpy.random import Generator, default_rng

from moptipy.algorithms.so.fitness import Fitness, FRecord, check_fitness
from moptipy.api.objective import Objective
from moptipy.api.operators import Op0
from moptipy.api.space import Space
from moptipy.tests.component import validate_component
from moptipy.utils.types import type_error


class _FRecord(FRecord):
    """The internal F-record."""

    def __init__(self, x, z):
        """Initialize."""
        super().__init__(x, inf)
        #: the internal id
        self.z: Final[int] = z


def validate_fitness(fitness: Fitness,
                     objective: Objective,
                     space: Space,
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
    pop1: Final[List[_FRecord]] = []
    pop2: Final[List[_FRecord]] = []
    for i in range(int(1 + random.integers(48))):
        fr: _FRecord = _FRecord(space.create(), i)
        op0.op0(random, fr.x)
        fr.f = objective.evaluate(fr.x)
        fr.it = int(random.integers(1, 20))
        if fr.v != inf:
            raise ValueError(f"v = {fr.v}, should be inf")
        pop1.append(fr)
        fr2: _FRecord = _FRecord(fr.x, fr.z)
        fr2.f = fr.f
        fr2.it = fr.it
        pop2.append(fr2)

    for k in range(6):

        if k >= 3:
            # make all records identical
            fr0 = pop1[0]
            for i, fr in enumerate(pop1):
                fr = _FRecord(fr0.x, i)
                fr.f = fr0.f
                fr.it = fr0.it
                fr.v = fr0.v
                pop1[i] = fr
                fr = _FRecord(fr0.x, i)
                fr.f = fr0.f
                fr.it = fr0.it
                fr.v = fr0.v
                pop2[i] = fr
        if k in (2, 4):
            for fr in pop1:
                if random.integers(2) <= 0:
                    fr.v = inf if random.integers(2) <= 0 else -inf

        fitness.assign_fitness(cast(List[FRecord], pop1), random)
        pop1.sort(key=lambda r: r.z)

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
            if not isinstance(fr.v, (int, float)):
                raise type_error(fr.v, "rec.v", (int, float))
            if not isfinite(fr.v):
                raise ValueError(f"rec.v should be finite, but is {fr.v}")
