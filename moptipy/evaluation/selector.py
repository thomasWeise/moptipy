"""
A tool for selecting a consistent subset of data from partial experiments.

When we have partial experimental data, maybe collected from experiments that
are still ongoing, we want to still evaluate them in some consistent way. The
right method for doing this could be to select a subset of that data that is
consistent, i.e., a subset where the algorithms have the same number of runs
on the instances using the same seeds. The function :func:`select_consistent`
offered by this module provides the functionality to make such a selection.
It may be a bit slow, but hopefully it will pick the largest possible
consistent sub-selection or, at least, get close to it.

The current method to select the data is rather heuristic. It always begins
with the full set of data and aims to delete the element that will cause the
least other deletions down the road, until we arrive in a consistent state.
I strongly suspect that doing this perfectly would be NP-hard, so we cannot
implement this. Instead, we use different heuristics and then pick the best
result.
"""

from collections import Counter
from operator import itemgetter
from typing import Any, Callable, Final, Iterable, TypeVar

from pycommons.io.console import logger
from pycommons.types import type_error

from moptipy.evaluation.base import PerRunData

#: the type variable for the selector routine
T = TypeVar("T", bound=PerRunData)

#: the algorithm key
KEY_ALGORITHM: Final[int] = 0
#: the instance key
KEY_INSTANCE: Final[int] = KEY_ALGORITHM + 1
#: the encoding key
KEY_ENCODING: Final[int] = KEY_INSTANCE + 1
#: the objective key
KEY_OBJECTIVE: Final[int] = KEY_ENCODING + 1
#: the seed key
KEY_SEED: Final[int] = KEY_OBJECTIVE + 1
#: the number of keys
TOTAL_KEYS: Final[int] = KEY_SEED + 1


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


def __ret(source: list, expected: int, log: bool) -> list:
    """
    Prepare a source list for return.

    :param source: the list
    :param expected: the number of expected records
    :param log: shall we log information
    :returns: the result
    """
    for i, er in enumerate(source):
        source[i] = er[1]
    if list.__len__(source) != expected:
        raise ValueError("Inconsistent list length!")
    if log:
        logger(f"Now returning {expected} records of data.")
    return source  # type: ignore


def __scorer_tired(d: tuple, scores: tuple[Counter, ...], _) -> list[int]:
    """
    Score based on the tired score.

    :param d: the tuple to score
    :param scores: the scores
    :returns: the score
    """
    return sorted(scores[i][t] for i, t in enumerate(d))


def __scorer_normalized_tired(d: tuple, scores: tuple[Counter, ...],
                              total: tuple[int, ...]) -> list[float]:
    """
    Score based on the tired score.

    :param d: the tuple to score
    :param scores: the scores
    :param total: the total of the scores
    :returns: the score
    """
    return sorted(
        scores[i][t] / total[i] for i, t in enumerate(d) if total[i] > 0)


def __scorer_sum(d: tuple, scores: tuple[Counter, ...], _) -> int:
    """
    Score based on the sum score.

    :param d: the tuple to score
    :param scores: the scores
    :returns: the score
    """
    return sum(scores[i][t] for i, t in enumerate(d))


def __scorer_normalized_sum(d: tuple, scores: tuple[Counter, ...],
                            total: tuple[int, ...]) -> float:
    """
    Score based on the sum score.

    :param d: the tuple to score
    :param scores: the scores
    :param total: the total of the scores
    :returns: the score
    """
    return sum(
        scores[i][t] / total[i] for i, t in enumerate(d) if total[i] > 0)


def __scorer_product(d: tuple, scores: tuple[Counter, ...], _) -> int:
    """
    Score based on the product score.

    :param d: the tuple to score
    :param scores: the scores
    :returns: the score
    """
    result: int = 1
    for i, t in enumerate(d):
        result *= scores[i][t]
    return result


def __scorer_normalized_min(d: tuple, scores: tuple[Counter, ...],
                            total: tuple[int, ...]) -> float:
    """
    Score based on the normalized minimum score.

    :param d: the tuple to score
    :param scores: the scores
    :param total: the total of the scores
    :returns: the score
    """
    return min(
        scores[i][t] / total[i] for i, t in enumerate(d) if total[i] > 0)


def __not(_: tuple[Counter, ...]) -> int:
    """
    Do nothing, return a placeholder.

    :returns -1: always
    """
    return -1


def __total(scores: tuple[Counter, ...]) -> tuple[int, ...] | None:
    """
    Get the total scores.

    :param scores: the scores
    :returns: the total scores
    """
    result: tuple[int, ...] = tuple(
        t.total() if len(t) > 1 else -1 for t in scores)
    return None if (tuple.__len__(result) < 1) or (
        max(result) <= 0) else result


#: the scorers
__SCORERS: Final[tuple[tuple[str, Callable[[
    tuple, tuple[Counter, ...], Any], Any], Callable[[
        tuple[Counter, ...]], Any]], ...]] = (
    ("tired", __scorer_tired, __not),
    ("normalized_tired", __scorer_normalized_tired, __total),
    ("sum", __scorer_sum, __not),
    ("normalized_sum", __scorer_normalized_sum, __total),
    ("product", __scorer_product, __total),
    ("normalized_min", __scorer_normalized_min, __total),
)


def select_consistent(data: Iterable[T], log: bool = True,
                      thorough: bool = True) -> list[T]:
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
    :param thorough: use the slower method which may give us more data
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
    ['a1/i1/o1/e1/2', 'a1/i2/o1/e1/2', 'a2/i1/o1/e1/2', 'a2/i2/o1/e1/2']

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
    data should be an instance of typing.Iterable but is int, namely 1.

    >>> try:
    ...     select_consistent((a2i1o1e1s2, a2i1o1e1s3), 3)
    ... except TypeError as te:
    ...     print(te)
    log should be an instance of bool but is int, namely 3.

    >>> try:
    ...     select_consistent({234})
    ... except TypeError as te:
    ...     print(te)
    dataElement should be an instance of moptipy.evaluation.base.PerRunData \
but is int, namely 234.

    >>> try:
    ...     select_consistent((a2i1o1e1s2, a2i1o1e1s3), True, 4)
    ... except TypeError as te:
    ...     print(te)
    thorough should be an instance of bool but is int, namely 4.
    """
    if not isinstance(data, Iterable):
        raise type_error(data, "data", Iterable)
    if not isinstance(log, bool):
        raise type_error(log, "log", bool)
    if not isinstance(thorough, bool):
        raise type_error(thorough, "thorough", bool)

    # We obtain a sorted list of the data in order to make the results
    # consistent regardless of the order in which the data comes in.
    # The data is only sorted by its features and not by any other information
    # attached to it.
    dataset: Final[list[list]] = [[__data_tup(t), t, 0] for t in data]
    dataset.sort(key=itemgetter(0))
    dataset_size: int = list.__len__(dataset)
    if log:
        logger(f"Found {dataset_size} records of data.")

    set_l: Final[int] = set.__len__({x[0] for x in dataset})
    if set_l != dataset_size:
        raise ValueError(
            f"Found {dataset_size} records but only {set_l} different keys!")

    # Compute the scores: count how often each algorithm, instance,
    # objective, encoding, and (instance, seed) combination are
    # encountered.
    key_tuple_len: Final[int] = tuple.__len__(dataset[0][0])
    if key_tuple_len != TOTAL_KEYS:
        raise ValueError("Unexpected error!")

    # the map for the score calculations
    scores: Final[tuple[Counter, ...]] = tuple(
        Counter() for _ in range(key_tuple_len))

    # the final result
    best_length: int = -1
    best_data: list[list] | None = None

    for scorer_name, scorer, scorer_total in __SCORERS if thorough else (
            __SCORERS[0], ):
        logger(f"Now scoring according to the {scorer_name!r} method.")
        # compute the original item scores
        for sc in scores:
            sc.clear()

        source: list[list] = dataset.copy()
        count: int = dataset_size
        for er in source:
            __score_inc(scores, er[0])

        changed: bool = True
        while changed:
            changed = False

            # Find the setups with maximum and minimum scores.
            min_score: Any = None
            max_score: Any = None
            norm: Any = scorer_total(scores)

            if norm is None:  # cannot proceed
                if log:
                    logger(f"Method {scorer_name!r} is not applicable.")
                count = -1  # we can do nothing
                break

            for er in source:
                score = scorer(er[0], scores, norm)
                er[2] = score
                if (min_score is None) or (min_score > score):
                    min_score = score
                if (max_score is None) or (max_score < score):
                    max_score = score
            del norm

            if min_score < max_score:  # Some setups have lower scores.
                del_count: int = 0  # We will delete them.
                for i in range(count - 1, -1, -1):
                    er = source[i]
                    if er[2] <= min_score:
                        del source[i]
                        __score_dec(scores, er[0])
                        del_count += 1
                if del_count <= 0:
                    raise ValueError(
                        f"Did not delete anything under {scorer_name!r}?")

                new_count: int = list.__len__(source)
                if new_count != (count - del_count):
                    raise ValueError("List inconsistent after deletion "
                                     f"under {scorer_name!r}?")
                if log:
                    logger(
                        f"Deleted {del_count} of the {count} records because "
                        f"their score was {min_score} while the maximum score"
                        f" was {max_score}. Retained {new_count} records "
                        f"under {scorer_name!r}.")
                count = new_count
                changed = True
                continue

            if log:
                logger(f"All setups now have the same score {max_score} under"
                       f" {scorer_name!r}.")
            if count <= best_length:
                if log:
                    logger(f"We now only have {count} setups, which means we "
                           "cannot get better than the current best set with "
                           f"{best_length} setups, so we quit after score-"
                           f"based cleaning under {scorer_name!r}.")
                count = -1
                break

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
                if (v != KEY_SEED) and (len(s) > 1))

            if tuple.__len__(keys) > 1:
                if log:
                    logger("Now checking for inconsistencies in "
                           "algorithm/instance/objective/encoding under"
                           f" {scorer_name!r}.")
                # Only if there are different values in at least one tuple
                # dimension, we need to check for strange situations.

                # For each of (instance, algorithm, encoding, objective), we
                # must have the same number of other records.
                # If not, then we have some strange symmetric situation that
                # needs to be solved.
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

                    # We just need to delete one random element. This will
                    # break the symmetry
                    if changed:
                        if log:
                            logger(
                                f"Deleting one of the {count} elements to "
                                "break the erroneous symmetry under "
                                f"{scorer_name}.")
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
                    logger("No inconsistencies in algorithm/instance/objecti"
                           f"ve/encoding found under {scorer_name!r}.")
            elif log:
                logger("No inconsistencies in algorithm/instance/objective/"
                       f"encoding possible under {scorer_name!r}.")
            if count <= best_length:
                if log:
                    logger(f"We now only have {count} setups, which means we "
                           "cannot get better than the current best set with "
                           f"{best_length} setups, so we quit after algorithm"
                           "/instance/objective/encoding cleaning under "
                           f"{scorer_name!r}.")
                count = -1
                break

            # If we get here, the only problem left could be if algorithms
            # have different seeds for the same instances. We thus need to
            # check that for each instance, all the seeds are the same.
            # Notice that such inconsistencies can only occur if different
            # seeds occurred exactly as same as often.
            if log:
                logger("Now checking for inconsistencies in instance "
                       f"seeds under {scorer_name!r}.")
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

            max_seed_insts: set[str] = set()
            max_seeds: int = -1
            min_seed_inst: str | None = None
            min_seeds: int = -1
            must_delete_from_insts: set[str] = set()
            for instance, inst_setups in seeds.items():
                kvx: set[int] | None = None
                for setup_seeds in inst_setups.values():
                    if kvx is None:
                        kvx = setup_seeds
                        seed_cnt: int = set.__len__(setup_seeds)
                        if seed_cnt >= max_seeds:
                            if seed_cnt > max_seeds:
                                max_seed_insts.clear()
                                max_seeds = seed_cnt
                            max_seed_insts.add(instance)
                        if (seed_cnt < min_seeds) or (min_seed_inst is None):
                            min_seeds = seed_cnt
                            min_seed_inst = instance
                    elif kvx != setup_seeds:
                        # We got a symmetric inconsistency
                        must_delete_from_insts.add(instance)
                        break
            if min_seeds < max_seeds:
                must_delete_from_insts.update(max_seed_insts)
            del max_seed_insts

            del_count = set.__len__(must_delete_from_insts)
            if del_count > 0:
                if log:
                    logger("Must delete one record from all of "
                           f"{must_delete_from_insts!r}.")
                for i in range(count - 1, -1, -1):
                    er = source[i]
                    instance = er[1].instance
                    if instance in must_delete_from_insts:
                        must_delete_from_insts.remove(instance)
                        del source[i]
                        __score_dec(scores, er[0])
                        changed = True
                        if set.__len__(must_delete_from_insts) <= 0:
                            break
                new_count = list.__len__(source)
                if new_count != (count - del_count):
                    raise ValueError("Error when deleting instances "
                                     f"under {scorer_name!r}.")
                count = new_count
                if not changed:
                    raise ValueError(
                        f"Seeds inconsistent under {scorer_name!r}.")
            del must_delete_from_insts

            del seeds
            if changed:
                if count <= best_length:
                    if log:
                        logger(f"We now only have {count} setups, which "
                               "means we cannot get better than the current "
                               f"best set with {best_length} setups, so we "
                               "quit after seed-based cleaning under "
                               f"{scorer_name!r}.")
                    count = -1
                    break
            elif log:
                logger(f"No seed inconsistencies under {scorer_name!r}.")
            # There should not be any problems left, but we need to check
            # again if something has changed.

        if count <= 0:
            continue  # We can do nothing here

        if count > best_length:
            logger(f"Method {scorer_name!r} yields {count} records, "
                   "which is the new best.")
            best_length = count
            best_data = source
            if best_length >= dataset_size:
                logger(f"Included all data under {scorer_name!r}, "
                       "so we can stop.")
                break
        elif log:
            logger(f"Method {scorer_name!r} yields {count} records, "
                   "which is not better than the current "
                   f"best {best_length}.")

    # We are finally finished. The scores are no longer needed.
    del scores
    if best_length <= 0:
        raise ValueError("No data found at all?")
    return __ret(best_data, best_length, log)
