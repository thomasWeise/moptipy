"""Validate selection algorithms."""

from math import inf
from typing import Final, Iterable

from numpy.random import Generator, default_rng

from moptipy.algorithms.modules.selection import (
    FitnessRecord,
    Selection,
    check_selection,
)
from moptipy.tests.component import validate_component
from moptipy.utils.types import type_error


class _FRecord(FitnessRecord):
    """The internal Fitness-record."""

    def __init__(self, tag: int) -> None:
        """Initialize."""
        #: the fitness
        self.fitness: int | float = inf
        #: the tag
        self.tag: Final[int] = tag

    def __str__(self) -> str:
        """Get the string describing this record."""
        return f"{self.tag}/{self.fitness}"


def __join(sets: Iterable[Iterable[int]],
           lower_limit: int | float = -inf,
           upper_limit: int | float = inf) -> Iterable[int]:
    """
    Joint iterables preserving unique values.

    :param sets: the iterables
    :param lower_limit: the lower limit
    :param upper_limit: the upper limit
    :returns: the joint iterable
    """
    if not isinstance(sets, Iterable):
        raise type_error(sets, "sets", Iterable)
    if not isinstance(lower_limit, (int, float)):
        raise type_error(lower_limit, "lower_limit", (int, float))
    if not isinstance(upper_limit, (int, float)):
        raise type_error(upper_limit, "upper_limit", (int, float))
    if upper_limit < lower_limit:
        raise ValueError(
            f"lower_limit={lower_limit} but upper_limit={upper_limit}")
    x: Final[set[int]] = set()
    for it in sets:
        if not isinstance(it, Iterable):
            raise type_error(it, "it", Iterable)
        x.update([int(i) for i in it if lower_limit <= i <= upper_limit])
    return list(x)


def validate_selection(selection: Selection,
                       without_replacement: bool = False,
                       lower_source_size_limit: int = 0,
                       upper_source_size_limit: int = 999999) -> None:
    """
    Validate a selection algorithm.

    :param selection: the selection algorithm
    :param without_replacement: is this selection algorithm without
        replacement, i.e., can it select each element at most once?
    :param lower_source_size_limit: the lower limit of the source size
    :param upper_source_size_limit: the upper limit for the source size
    """
    check_selection(selection)
    validate_component(selection)

    if not isinstance(without_replacement, bool):
        raise type_error(without_replacement, "without_replacement", bool)
    if not isinstance(lower_source_size_limit, int):
        raise type_error(lower_source_size_limit,
                         "lower_source_size_limit", int)
    if not 0 <= lower_source_size_limit < 1000:
        raise ValueError(f"lower_source_size_limit={lower_source_size_limit}")
    if not isinstance(upper_source_size_limit, int):
        raise type_error(upper_source_size_limit,
                         "upper_source_size_limit", int)
    if upper_source_size_limit < lower_source_size_limit:
        raise ValueError(
            f"upper_source_size_limit={upper_source_size_limit} while "
            f"lower_source_size_limit={lower_source_size_limit}")

    random: Final[Generator] = default_rng()
    source: Final[list[FitnessRecord]] = []
    copy: Final[dict[int, list]] = {}
    dest: Final[list[FitnessRecord]] = []
    tag: int = 0

    for source_size in __join([range(1, 10), [16, 32, 50, 101],
                               random.choice(100, 4, False),
                               [lower_source_size_limit]],
                              lower_limit=max(1, lower_source_size_limit),
                              upper_limit=upper_source_size_limit):
        for dest_size in __join([
            [1, 2, 3, source_size, 2 * source_size],
            random.choice(min(6, 4 * source_size),
                          min(6, 4 * source_size), False)],
                upper_limit=source_size if without_replacement else inf,
                lower_limit=1):

            source.clear()
            copy.clear()
            dest.clear()

            # choose the fitness function
            fit_choice = random.integers(3)
            if fit_choice == 0:
                def fitness(  # noqa
                        value=int(random.integers(-10, 10))):  # type: ignore
                    return value
            elif fit_choice == 1:
                def fitness(  # noqa
                        limit=random.integers(100) + 1):  # type: ignore
                    return int(random.integers(limit))
            else:
                def fitness():  # type: ignore # noqa
                    return float(random.uniform())

            for _ in range(source_size):
                tag += 1
                r1 = _FRecord(tag)
                r1.fitness = fitness()
                r2 = _FRecord(tag)
                r2.fitness = r1.fitness
                source.append(r1)
                copy[r2.tag] = [r2, 0, False]

            # perform the selection
            selection.select(source, dest.append, dest_size, random)

            if len(dest) != dest_size:
                raise ValueError(
                    f"expected {selection} to select {dest_size} elements "
                    f"from {source_size}, but got {len(dest)} instead")
            if len(source) != source_size:
                raise ValueError(
                    f"selection {selection} changed length {source_size} "
                    f"of source list to {len(source)}")
            for ele in dest:
                if not isinstance(ele, _FRecord):
                    raise type_error(ele, "element in dest", _FRecord)
                if ele.tag not in copy:
                    raise ValueError(
                        f"element with tag {ele.tag} does not exist in "
                        f"source but was selected by {selection}?")
                check = copy[ele.tag]
                if check[0].fitness != ele.fitness:
                    raise ValueError(
                        f"fitness of source element {check[0].fitness} has "
                        f"been changed to {ele.fitness} by {selection} "
                        "in dest")
                check[1] += 1
                if without_replacement and (check[1] > 1):
                    raise ValueError(
                        f"{selection} is without replacement, but selected "
                        f"element with tag {ele.tag} at least twice!")
            for ele in source:
                if not isinstance(ele, _FRecord):
                    raise type_error(ele, "element in source", _FRecord)
                if ele.tag not in copy:
                    raise ValueError(
                        f"element with tag {ele.tag} does not exist in "
                        f"source but was created by {selection}?")
                check = copy.get(ele.tag)
                if check[0].fitness != ele.fitness:
                    raise ValueError(
                        f"fitness of source element {check[0].fitness} has "
                        f"been changed to {ele.fitness} by {selection} "
                        "in source")
                if check[2]:
                    raise ValueError(
                        f"element with tag {ele.tag} has been replicated "
                        f"by {selection} in source?")
                check[2] = True
            for check in copy.values():
                if not check[2]:
                    raise ValueError(
                        f"element with tag {check[0].tag} somehow lost?")
