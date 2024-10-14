"""
A tool for selecting data.

When we have partial experimental data, maybe collected from experiments that
are still ongoing, we want to still evaluate them in some consistent way. The
right method for doing this could be to select a subset of that data that is
consistent, i.e., a subset where the algorithms have the same number of runs
on the instances using the same seeds. The function :func:`select_consistent`
offered by this module provides the functionality to make such a selection.
It may be a bit slow, but hopefully it will pick the largest possible
consistent sub-selection or, at least, get close to it.
"""

from collections import Counter
from typing import Any, Final, Iterable, TypeVar

from pycommons.io.console import logger
from pycommons.types import type_error

from moptipy.evaluation.base import PerRunData

#: the type variable for the selector routine
T = TypeVar("T", bound=PerRunData)


def __data_tup(d: PerRunData) -> tuple[str, str, str, str, tuple[str, int]]:
    """
    Get a raw data tuple for the given record.

    :param d: the data record
    :returns: the data tuple
    """
    if not isinstance(d, PerRunData):
        raise type_error(d, "dataElement", PerRunData)
    return d.algorithm, d.instance, d.encoding, d.objective, (
        d.instance, d.rand_seed)


def __score_inc(scores: tuple[Counter, ...], key: tuple[Any, ...]) -> None:
    """
    Increment the score of a given tuple.

    :param scores: the scores
    :param key: the key
    """
    for i, k in enumerate(key):
        scores[i][k] += 1


def __score_dec(scores: tuple[Counter, ...], key: tuple[Any, ...]) -> None:
    """
    Decrement the score of a given tuple.

    :param scores: the scores
    :param key: the key
    """
    for i, k in enumerate(key):
        scores[i][k] -= 1
        z: int = scores[i][k]
        if z <= 0:
            del scores[i][k]
            if z < 0:
                raise ValueError("Got negative score?")


def select_consistent(data: Iterable[T], log: bool = True) -> list[T]:
    """
    Select data such that the numbers of runs are consistent.

    The input is a set of data items which represent some records over the
    runs of algorithms on instances. It may be that not all algorithms have
    been applied to all instances. Maybe the number of runs is inconsistent
    over the algorithm-instance combinations, too. Maybe some algorithms
    have more runs on some instances. Maybe the runs are even different,
    it could be that some algorithms have runs for seed `A`, `B`, and `C` on
    instance `I`, while others have runs for seed `C` and `D`. This function
    is designed to retain only the runs with seed `C` in such a case. It may
    discard algorithms or instances or algorithm-instance-seed combinations
    in order to obtain a selection of data where all algorithms have been
    applied to all instances as same as often and using the same seeds.

    Now there are different ways to select such consistent subsets of a
    dataset. Of course, we want to select the data such that as much as
    possible of the data is retained and as little as possible is discarded.
    This may be a hard optimization problem in itself. Here, we offer a
    heuristic solution. Basically, we step-by-step try to cut away the
    setups that are covered by the least amount of runs. We keep repeating
    this until we arrive in a situation where all setups have the same
    amount of runs. We then check if there were some strange symmetries that
    still make the data inconsistent and, if we found some, try to delete
    one run to break the symmetries and then repeat the cleaning-up process.
    In the end, we should get a list of overall consistent data elements that
    can be used during a normal experiment evaluation procedure.

    This iterative process may be rather slow on larger datasets, but it is
    maybe the best approximation we can offer to retain as much data as
    possible.

    :param data: the source data
    :param log: shall we log the progress
    :return: a list with the selected data

    >>> def __p(x) -> str:
    ...     return (f"{x.algorithm}/{x.instance}/{x.objective}/{x.encoding}/"
    ...             f"{x.rand_seed}")

    >>> a1i1o1e1s1 = PerRunData("a1", "i1", "o1", "e1", 1)
    >>> a1i1o1e1s2 = PerRunData("a1", "i1", "o1", "e1", 2)
    >>> a1i1o1e1s3 = PerRunData("a1", "i1", "o1", "e1", 3)
    >>> a1i2o1e1s1 = PerRunData("a1", "i2", "o1", "e1", 1)
    >>> a1i2o1e1s2 = PerRunData("a1", "i2", "o1", "e1", 2)
    >>> a1i2o1e1s3 = PerRunData("a1", "i2", "o1", "e1", 3)
    >>> a2i1o1e1s1 = PerRunData("a2", "i1", "o1", "e1", 1)
    >>> a2i1o1e1s2 = PerRunData("a2", "i1", "o1", "e1", 2)
    >>> a2i1o1e1s3 = PerRunData("a2", "i1", "o1", "e1", 3)
    >>> a2i2o1e1s1 = PerRunData("a1", "i2", "o1", "e1", 1)
    >>> a2i2o1e1s2 = PerRunData("a2", "i2", "o1", "e1", 2)
    >>> a2i2o1e1s3 = PerRunData("a2", "i2", "o1", "e1", 3)

    >>> list(map(__p, select_consistent((
    ...     a1i1o1e1s1, a1i1o1e1s2, a1i1o1e1s3,
    ...     a1i2o1e1s1, a1i2o1e1s2, a1i2o1e1s3,
    ...     a2i1o1e1s1, a2i1o1e1s2,
    ...     a2i2o1e1s2, a2i2o1e1s3))))
    ['a1/i1/o1/e1/1', 'a1/i1/o1/e1/2', 'a1/i2/o1/e1/2', 'a1/i2/o1/e1/3',\
 'a2/i1/o1/e1/1', 'a2/i1/o1/e1/2', 'a2/i2/o1/e1/2', 'a2/i2/o1/e1/3']

    >>> list(map(__p, select_consistent((
    ...     a1i1o1e1s2, a1i1o1e1s3,
    ...     a1i2o1e1s1, a1i2o1e1s2, a1i2o1e1s3,
    ...     a2i1o1e1s1, a2i1o1e1s2,
    ...     a2i2o1e1s2, a2i2o1e1s3))))
    ['a1/i2/o1/e1/2', 'a1/i2/o1/e1/3', 'a2/i2/o1/e1/2', 'a2/i2/o1/e1/3']

    >>> list(map(__p, select_consistent((
    ...     a1i1o1e1s2, a1i1o1e1s3,
    ...     a1i2o1e1s1, a1i2o1e1s2, a1i2o1e1s3,
    ...     a2i1o1e1s1, a2i1o1e1s2,
    ...     a2i2o1e1s2))))
    ['a1/i2/o1/e1/1', 'a1/i2/o1/e1/2', 'a1/i2/o1/e1/3']

    >>> list(map(__p, select_consistent((
    ...     a1i1o1e1s1, a1i1o1e1s2, a1i1o1e1s3,
    ...     a2i2o1e1s1, a2i2o1e1s2, a2i2o1e1s3))))
    ['a1/i1/o1/e1/1', 'a1/i1/o1/e1/2', 'a1/i1/o1/e1/3']

    >>> list(map(__p, select_consistent((
    ...     a1i1o1e1s1, a1i1o1e1s2, a1i1o1e1s3,
    ...     a2i1o1e1s1, a2i1o1e1s2, a2i1o1e1s3))))
    ['a1/i1/o1/e1/1', 'a1/i1/o1/e1/2', 'a1/i1/o1/e1/3', \
'a2/i1/o1/e1/1', 'a2/i1/o1/e1/2', 'a2/i1/o1/e1/3']

    >>> list(map(__p, select_consistent((
    ...     a1i1o1e1s1, a1i1o1e1s2, a1i2o1e1s2, a1i2o1e1s3))))
    ['a1/i1/o1/e1/1', 'a1/i1/o1/e1/2', 'a1/i2/o1/e1/2', 'a1/i2/o1/e1/3']

    >>> list(map(__p, select_consistent((
    ...     a1i1o1e1s1, a1i1o1e1s2, a1i2o1e1s2))))
    ['a1/i1/o1/e1/1', 'a1/i1/o1/e1/2']

    >>> list(map(__p, select_consistent((
    ...     a1i1o1e1s1, a2i1o1e1s2))))
    ['a1/i1/o1/e1/1']

    >>> list(map(__p, select_consistent((
    ...     a1i1o1e1s1, a2i1o1e1s2, a2i1o1e1s3))))
    ['a2/i1/o1/e1/2', 'a2/i1/o1/e1/3']

    >>> try:
    ...     select_consistent((a1i1o1e1s1, a1i1o1e1s2, a1i2o1e1s2, a1i2o1e1s2))
    ... except ValueError as ve:
    ...     print(ve)
    Found 4 records but only 3 different keys!

    >>> try:
    ...     select_consistent(1)
    ... except TypeError as te:
    ...     print(te)
    data should be an instance of typing.Iterable but is int, namely '1'.

    >>> try:
    ...     select_consistent((a2i1o1e1s2, a2i1o1e1s3), 3)
    ... except TypeError as te:
    ...     print(te)
    log should be an instance of bool but is int, namely '3'.

    >>> try:
    ...     select_consistent({234})
    ... except TypeError as te:
    ...     print(te)
    dataElement should be an instance of moptipy.evaluation.base.PerRunData \
but is int, namely '234'.
    """
    if not isinstance(data, Iterable):
        raise type_error(data, "data", Iterable)
    if not isinstance(log, bool):
        raise type_error(log, "log", bool)

    # We obtain a sorted list of the data in order to make the results
    # consistent regardless of the order in which the data comes in.
    # The data is only sorted by its features and not by any other information
    # attached to it.
    source: Final[list[list]] = [[__data_tup(t), t, 0] for t in data]
    source.sort(key=lambda x: x[0])
    count: int = list.__len__(source)
    if log:
        logger(f"Found {count} records of data.")

    set_l: Final[int] = set.__len__({x[0] for x in source})
    if set_l != count:
        raise ValueError(
            f"Found {count} records but only {set_l} different keys!")

    changed: bool = True

    # Compute the scores: count how often each algorithm, instance,
    # objective, encoding, and (instance, seed) combination are
    # encountered.
    key_tuple_len: Final[int] = tuple.__len__(source[0][0])
    if key_tuple_len != 5:
        raise ValueError("Unexpected error!")
    scores: tuple[Counter, ...] = tuple(
        Counter() for _ in range(key_tuple_len))
    for er in source:
        __score_inc(scores, er[0])

    while changed:
        changed = False

        # Find the setups with maximum and minimum scores.
        min_score: int = -1
        max_score: int = -1
        for er in source:
            score: int = 1
            for i, t in enumerate(er[0]):
                score *= scores[i][t]
            er[2] = score
            min_score = score if min_score < 0 else min(score, min_score)
            max_score = max(score, max_score)

        if min_score < max_score:  # At least some setups have lower scores.
            del_count: int = 0  # We will delete them.
            for i in range(count - 1, -1, -1):
                er = source[i]
                if er[2] <= min_score:
                    del source[i]
                    __score_dec(scores, er[0])
                    del_count += 1
            if del_count <= 0:
                raise ValueError("Did not delete anything?")

            new_count: int = list.__len__(source)
            if new_count != (count - del_count):
                raise ValueError("List inconsistent after deletion?")
            if log:
                logger(f"Deleted {del_count} of the {count} records because "
                       f"their score was {min_score} while the maximum score "
                       f"was {max_score}. Retained {new_count} records.")
            count = new_count
            changed = True
            continue

        if log:
            logger(f"All setups now have the same score {max_score}.")

        # If we get here, all elements have the same score.
        # This means that we are basically done.
        #
        # However, this may also happen if a very odd division exists in
        # the data. Maybe we have one algorithm that was applied to one
        # instance ten times and another algorithm applied to another
        # instance ten times. This data would still be inconsistent, as it
        # does not allow for any comparison.

        # Compute the different values for everything except the seeds.
        keys: tuple[int, ...] = tuple(
            v for v, s in enumerate(scores)
            if (v < (key_tuple_len - 1)) and (len(s) > 1))

        if tuple.__len__(keys) > 1:
            if log:
                logger("Now checking for inconsistencies in "
                       "algorithm/instance/objective/encoding.")
            # Only if there are different values in at least one tuple
            # dimension, we need to check for strange situations.

            # For each of (instance, algorithm, encoding, objective), we must
            # have the same number of other records.
            # If not, then we have some strange symmetric situation that needs
            # to be solved.
            per_value: dict[Any, set[Any]] = {}
            last: set[Any] | None = None
            for key in keys:
                other_keys: tuple[int, ...] = tuple(
                    kk for kk in keys if kk != key)
                for er in source:
                    kv = er[0][key]
                    other = tuple(er[0][ok] for ok in other_keys)
                    if kv in per_value:
                        per_value[kv].add(other)
                    else:
                        per_value[kv] = last = {other}
                for v in per_value.values():
                    if v != last:
                        changed = True
                        break
                per_value.clear()

                # We just need to delete one random element. This will break
                # the symmetry
                if changed:
                    if log:
                        logger(f"Deleting one of the {count} elements to "
                               "break the erroneous symmetry.")
                    er = source[-1]
                    del source[-1]
                    __score_dec(scores, er[0])
                    count -= 1
                    break
            del per_value
            del keys
            if changed:
                continue

            if log:
                logger("No inconsistencies in algorithm/instance/objective/"
                       "encoding found.")
        elif log:
            logger("No inconsistencies in algorithm/instance/objective/"
                   "encoding possible.")

        # If we get here, the only problem left could be if algorithms have
        # different seeds for the same instances. We thus need to check that
        # for each instance, all the seeds are the same.
        # Notice that such inconsistencies can only occur if different seeds
        # occurred exactly as same as often.
        if log:
            logger("No checking for inconsistencies in instance seeds.")
        seeds: dict[str, dict[tuple[str, str, str], set[int]]] = {}
        for er in source:
            inst: str = er[1].instance
            cur: dict[tuple[str, str, str], set[int]]
            if inst in seeds:
                cur = seeds[inst]
            else:
                seeds[inst] = cur = {}
            kx: tuple[str, str, str] = (
                er[1].algorithm, er[1].objective, er[1].encoding)
            if kx in cur:
                cur[kx].add(er[1].rand_seed)
            else:
                cur[kx] = {er[1].rand_seed}

        for instance, inst_setups in seeds.items():
            kvx: set[int] | None = None
            for setup_seeds in inst_setups.values():
                if kvx is None:
                    kvx = setup_seeds
                elif kvx != setup_seeds:
                    # We got a symmetric inconsistency
                    for i in range(count - 1, -1, -1):
                        er = source[i]
                        if er[1].instance == instance:
                            if log:
                                logger(
                                    f"Deleting one of the {count} elements to"
                                    f" break an erroneous seed symmetry.")
                            del source[i]
                            __score_dec(scores, er[0])
                            count -= 1
                            changed = True
                            break
                    if not changed:
                        raise ValueError("Seeds inconsistent.")
                    break
            if changed:  # leave instance checking loop
                break
        del seeds
        # There should not be any problems left, but we need to check again.

    # We are finally finished. The scores are no longer needed.
    del scores

    for i, er in enumerate(source):
        source[i] = er[1]
    if list.__len__(source) != count:
        raise ValueError("Inconsistent list length!")

    if log:
        logger(f"Now returning {count} records of data.")
    return source  # type: ignore
