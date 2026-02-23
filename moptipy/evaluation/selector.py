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

Our :func:`select_consistent` tries to find such a consistent dataset using
different :class:`Selector` methods that are passed to it as second parameter.

In the most basic and defaul setting, :const:`SELECTOR_SAME_RUNS_FOR_ALL`, it
simply retains the exactly same number of runs for all
instance/algorithm/objective/encoding combinations, making sure that the same
seeds are used for a given instance.
This can be done rather quickly, but it may not yield the largest possible
set of consistent runs.
It will also fail if there are some combinations that have runs with mutually
distinct seeds.

In this case, using :const:`SELECTOR_MAX_RUNS_SIMPLE` might be successful, as
it is a more heuristic approach.
The heuristic methods implemented here always begin with the full set of data
and aim to delete the element that will cause the least other deletions down
the road, until we arrive in a consistent state.
I strongly suspect that doing this perfectly would be NP-hard, so we cannot
implement this in a perfect way. Instead, we use different heuristics and then
pick the best result.
:const:`SELECTOR_MAX_RUNS` uses more heuristics, which means that it will be
slower, but may have a bigger chance to yield a consistent experiment.
"""

from typing import Any, Final, Iterable, TypeVar

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


class Selector:
    """
    The base class for selectors.

    Each selector offers a routine that can select a subset of runs from a
    given description. For the selection process, all instances, objective
    functions, encodings, and algorithms as well as (instance-seed)
    combinations are replaced with integer numbers identifying them.

    The data that is passed to the method :meth:`select` is already purified.
    Algorithm, instance, objective function, and encoding names are replaced
    with consecutive integers. Instance/seed combinations are also replaced
    with such integer values.
    Each run of the original experiment is thus identified by a five-tuple of
    integer values.
    """

    def select(self,
               dims: tuple[int, int, int, int, int],
               keys: tuple[tuple[int, int, int, int, int], ...],
               best_length: int) -> list[tuple[
            int, int, int, int, int]] | None:
        """
        Perform the dataset selection.

        The `keys` array is an immutable list of keys. Each key uniquely
        identifies one run in the experiment. It is a tuple of five integer
        values:

        0. The value at index :const:`KEY_ALGORITHM` identifies the
           algorithm used in the run,
        1. the value at index :const:`KEY_INSTANCE` identifies the instance,
        2. the value at index :const:`KEY_OBJECTIVE` identifies the objective
           function,
        3. the value at index :const:`KEY_ENCODING` identifies the encoding,
           and
        4. the value at index :const:`KEY_SEED` identifies the instance-seed
           combination, meaning that each seed on each instance maps to a
           unique value here.

        The number of values in each key dimension is given in the array
        `dims`. The instance identifiers, for example, are all in the interval
        `0..dims[KEY_INSTANCE]-1`. Because of this, you can use the key values
        as indexes into consecutive lists to keep track of counts, if you want
        to.

        The parameter `best_length` provides the largest number of consistent
        runs discovered by any selection method so far. It is `-1` for the
        first selector that is attempted.
        The selection routines returns a list of selected run keys, or `None`
        if this selector could not surpass `best_length`.

        :param dims: the number of different values per key dimension
        :param keys: the data keys to select from
        :param best_length: the largest number of consistent runs discovered
            by any previously applied selector.
        :return: the selected data
        """
        raise NotImplementedError


def _count_inc(scores: list[list[int]], key: tuple[int, ...]) -> None:
    """
    Increment the score of a given tuple.

    :param scores: the scores
    :param key: the key
    """
    for i, k in enumerate(key):
        scores[i][k] += 1


def _count_dec(scores: list[list[int]], key: tuple[int, ...]) -> None:
    """
    Decrement the score of a given tuple.

    :param scores: the scores
    :param key: the key
    """
    for i, k in enumerate(key):
        scores[i][k] -= 1


class HeuristicSelector(Selector):
    """A heuristic selector."""

    def weight(self, counts: list[list[int]]) -> Any:  # pylint: disable=W0613
        """
        Compute a weight for the scores.

        This function can be overwritten to compute the total score.

        :param counts: the number of currently selected runs for each
            dimension and each dimension value
        :return: a weight factor, that may be ignored by some methods
        """
        return 1

    def score(self, element: tuple[int, ...],  # pylint: disable=W0613
              counts: list[list[int]],  # pylint: disable=W0613
              weights: Any) -> Any:  # pylint: disable=W0613
        """
        Score a given element.

        :param element: the tuple to score
        :param counts: the number of currently selected runs for each
            dimension and each dimension value
        :param weights: the weights, which may sometimes be ignored
        :returns: the score
        """
        return None

    def select(self,
               dims: tuple[int, int, int, int, int],
               keys: tuple[tuple[int, int, int, int, int], ...],
               best_length: int) -> list[tuple[
            int, int, int, int, int]] | None:
        """
        Perform the heuristic dataset selection.

        Heuristic selectors attempt to step-by-step delete keys that lead to
        the fewest dropped instances, algorithms, objectives, and encodings.

        :param dims: the number of different values per key dimension
        :param keys: the data keys to select from
        :param best_length: the length of the best-so-far set of runs,
            will be `-1` if no such best exists
        :return: the selected data
        """
        # the counters for the score calculations
        source: Final[list[list]] = [[k, 0] for k in keys]
        dataset_size: Final[int] = list.__len__(source)
        scorer_name: Final[str] = str(self)

        # compute the original item scores
        counts: list[list[int]] = [[0] * i for i in dims]
        count: int = dataset_size
        for er in source:
            _count_inc(counts, er[0])

        changed: bool = True
        while changed:
            changed = False

            # Find the setups with maximum and minimum scores.
            min_score: Any = None
            max_score: Any = None
            weights: Any = self.weight(counts)

            if weights is None:  # cannot proceed
                logger(f"Method {scorer_name!r} is not applicable.")
                return None

            for er in source:
                er[1] = score = self.score(  # pylint: disable=E1128
                    er[0], counts, weights)
                if score is not None:
                    if (min_score is None) or (min_score > score):
                        min_score = score
                    if (max_score is None) or (max_score < score):
                        max_score = score
            del weights
            if min_score is None:
                logger(f"{scorer_name!r} failed.")
                return None
            if min_score < max_score:  # Some setups have lower scores.
                del_count: int = 0  # We will delete them.
                for i in range(count - 1, -1, -1):
                    er = source[i]
                    score = er[1]
                    if (score is None) or (score <= min_score):
                        del source[i]
                        _count_dec(counts, er[0])
                        del_count += 1
                if del_count <= 0:
                    raise ValueError(
                        f"Did not delete anything under {scorer_name!r}?")

                new_count: int = list.__len__(source)
                if new_count != (count - del_count):
                    raise ValueError("List inconsistent after deletion "
                                     f"under {scorer_name!r}?")
                logger(f"Deleted {del_count} of the {count} records because "
                       f"their score was {min_score}. Retained {new_count} "
                       f"records under {scorer_name!r}.")
                count = new_count
                changed = True
                continue

            logger(f"All setups now have the same score {max_score} under"
                   f" {scorer_name!r}.")
            if count <= best_length:
                logger(f"We now only have {count} setups, which means we "
                       "cannot get better than the current best set with "
                       f"{best_length} setups, so we quit.")
                return None

            # If we get here, all elements have the same score.
            # This means that we are basically done.
            #
            # However, this may also happen if a very odd division exists in
            # the data. Maybe we have one algorithm that was applied to one
            # instance ten times and another algorithm applied to another
            # instance ten times. This data would still be inconsistent, as it
            # does not allow for any comparison.

            # Compute the different values for everything except the seeds.
            usekeys: tuple[int, ...] = tuple(
                v for v, s in enumerate(counts)
                if (v != KEY_SEED) and (len(s) > 1))

            if tuple.__len__(usekeys) > 1:
                logger("Now checking for inconsistencies in "
                       "algorithm/instance/objective/encoding under"
                       f" {scorer_name!r}.")
                # Only if there are different values in at least one tuple
                # dimension, we need to check for strange situations.

                # For each of (instance, algorithm, encoding, objective), we
                # must have the same number of other records.
                # If not, then we have some strange symmetric situation that
                # needs to be solved.
                per_value: dict[int, set[Any]] = {}
                last: set[Any] | None = None
                for key in usekeys:
                    other_keys: tuple[int, ...] = tuple(
                        kk for kk in usekeys if kk != key)
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
                        logger(
                            f"Deleting one of the {count} elements to "
                            "break the erroneous symmetry under "
                            f"{scorer_name}.")
                        er = source[-1]
                        del source[-1]
                        _count_dec(counts, er[0])
                        count -= 1
                        break
                del per_value
                del usekeys
                if changed:
                    continue

            if count <= best_length:
                logger(f"We now only have {count} setups, which means we "
                       "cannot get better than the current best set with "
                       f"{best_length} setups, so we quit.")
                return None

            # If we get here, the only problem left could be if algorithms
            # have different seeds for the same instances. We thus need to
            # check that for each instance, all the seeds are the same.
            # Notice that such inconsistencies can only occur if different
            # seeds occurred exactly as same as often.
            logger("Now checking for inconsistencies in instance "
                   f"seeds under {scorer_name!r}.")
            seeds: dict[int, dict[tuple[int, int, int], set[int]]] = {}
            for er in source:
                inst: int = er[0][KEY_INSTANCE]
                cur: dict[tuple[int, int, int], set[int]]
                if inst in seeds:
                    cur = seeds[inst]
                else:
                    seeds[inst] = cur = {}
                kx: tuple[int, int, int] = (
                    er[0][KEY_ALGORITHM], er[0][KEY_OBJECTIVE],
                    er[0][KEY_ENCODING])
                if kx in cur:
                    cur[kx].add(er[0][KEY_SEED])
                else:
                    cur[kx] = {er[0][KEY_SEED]}

            max_seed_insts: set[int] = set()
            max_seeds: int = -1
            min_seed_inst: int | None = None
            min_seeds: int = -1
            must_delete_from_insts: set[int] = set()
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
                logger("Must delete one record from all of "
                       f"{must_delete_from_insts!r}.")
                for i in range(count - 1, -1, -1):
                    er = source[i]
                    instance = er[0][KEY_INSTANCE]
                    if instance in must_delete_from_insts:
                        must_delete_from_insts.remove(instance)
                        del source[i]
                        _count_dec(counts, er[0])
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
            if count <= best_length:
                logger(f"We now only have {count} setups, which "
                       "means we cannot get better than the current "
                       f"best set with {best_length} setups, so we "
                       "quit.")
                return None
            # There should not be any problems left, but we need to check
            # again if something has changed.

        if count <= best_length:
            logger(f"We now only have {count} setups, which means we "
                   "cannot get better than the current best set with "
                   f"{best_length} setups, so we quit..")
            return None  # We can do nothing here
        # We are finally finished. The scores are no longer needed.
        del counts
        if count <= 0:
            raise ValueError("No data found at all?")
        return [b[0] for b in source]


class __HeuristicSelectorTiered(HeuristicSelector):
    """A tiered heuristic selector."""

    def score(self, element: tuple[int, ...], counts: list[list[int]],
              _: Any) -> list[int]:
        """
        Score a given element.

        :param element: the tuple to score
        :param counts: the element counts per dimension
        :param _: the weights, which may sometimes be ignored
        :returns: the score
        """
        return sorted(counts[i][t] for i, t in enumerate(element))

    def __str__(self) -> str:
        """
        Get the selector name.

        :returns "tiered": always
        """
        return "tiered"


class __HeuristicSelectorTieredReverse(HeuristicSelector):
    """A reverse tiered heuristic selector."""

    def score(self, element: tuple[int, ...], counts: list[list[int]],
              _: Any) -> list[int]:
        """
        Score a given element.

        :param element: the tuple to score
        :param counts: the element counts per dimension
        :param _: the weights, which may sometimes be ignored
        :returns: the score
        """
        return sorted((counts[i][t] for i, t in enumerate(element)),
                      reverse=True)

    def __str__(self) -> str:
        """
        Get the selector name.

        :returns "tieredReverse": always
        """
        return "tieredReverse"


class __HeuristicSelectorSum(HeuristicSelector):
    """A sum heuristic selector."""

    def score(self, element: tuple[int, ...], counts: list[list[int]],
              _: Any) -> int:
        """
        Score a given element.

        :param element: the tuple to score
        :param counts: the element counts per dimension
        :param _: the weights, which may sometimes be ignored
        :returns: the score
        """
        return sum(counts[i][t] for i, t in enumerate(element))

    def __str__(self) -> str:
        """
        Get the selector name.

        :returns "sum": always
        """
        return "sum"


class __HeuristicSelectorProduct(HeuristicSelector):
    """A product heuristic selector."""

    def score(self, element: tuple[int, ...], counts: list[list[int]],
              _: Any) -> int:
        """
        Score a given element.

        :param element: the tuple to score
        :param counts: the element counts per dimension
        :param _: the weights, which may sometimes be ignored
        :returns: the score
        """
        result: int = 1
        for i, t in enumerate(element):
            result *= counts[i][t]
        return result

    def __str__(self) -> str:
        """
        Get the selector name.

        :returns "product": always
        """
        return "product"


class __HeuristicSelectorMinimum(HeuristicSelector):
    """A minimum heuristic selector."""

    def score(self, element: tuple[int, ...], counts: list[list[int]],
              _: Any) -> int:
        """
        Score a given element.

        :param element: the tuple to score
        :param counts: the element counts per dimension
        :param _: the weights, which may sometimes be ignored
        :returns: the score
        """
        return min(element)

    def __str__(self) -> str:
        """
        Get the selector name.

        :returns "min": always
        """
        return "min"


class __NormalizedHeuristicSelector(HeuristicSelector):
    """A heuristic selector."""

    def weight(self, counts: list[list[int]]) -> list[int] | None:
        """
        Compute the total score.

        This function can be overwritten to compute the total score.

        :param counts: the counts
        :return: a weight factor, that may be ignored by some methods
        """
        result: list[int] = [sum(t) for t in counts]
        return None if (list.__len__(result) < 1) or (
            max(result) <= 0) else result


class __HeuristicSelectorNormTiered(__NormalizedHeuristicSelector):
    """A normalized tiered heuristic selector."""

    def score(self, element: tuple[int, ...], counts: list[list[int]],
              weights: Any) -> list[float]:
        """
        Score a given element.

        :param element: the tuple to score
        :param counts: the element counts per dimension
        :param weights: the weights
        :returns: the score
        """
        return sorted(counts[i][t] / weights[i] for i, t in enumerate(
            element) if weights[i] > 0)

    def __str__(self) -> str:
        """
        Get the selector name.

        :returns "normalizedTiered": always
        """
        return "normalizedTiered"


class __HeuristicSelectorNormTieredRev(__NormalizedHeuristicSelector):
    """A normalized tiered reverse heuristic selector."""

    def score(self, element: tuple[int, ...], counts: list[list[int]],
              weights: Any) -> list[float]:
        """
        Score a given element.

        :param element: the tuple to score
        :param counts: the element counts per dimension
        :param weights: the weights
        :returns: the score
        """
        return sorted((counts[i][t] / weights[i] for i, t in enumerate(
            element) if weights[i] > 0), reverse=True)

    def __str__(self) -> str:
        """
        Get the selector name.

        :returns "normalizedTieredReverse": always
        """
        return "normalizedTieredReverse"


class __HeuristicSelectorNormSum(__NormalizedHeuristicSelector):
    """A normalized sum heuristic selector."""

    def score(self, element: tuple[int, ...], counts: list[list[int]],
              weights: Any) -> float:
        """
        Score a given element.

        :param element: the tuple to score
        :param counts: the element counts per dimension
        :param weights: the weights
        :returns: the score
        """
        return sum(counts[i][t] / weights[i] for i, t in enumerate(
            element) if weights[i] > 0)

    def __str__(self) -> str:
        """
        Get the selector name.

        :returns "normalizedSum": always
        """
        return "normalizedSum"


class __SameNumberOfRuns(Selector):
    """A selector choosing the same runs for all instance/algorithm combos."""

    def select(self,
               dims: tuple[int, int, int, int, int],
               keys: tuple[tuple[int, int, int, int, int], ...],
               best_length: int) -> list[tuple[
            int, int, int, int, int]] | None:
        """
        Perform the dataset selection.

        :param dims: the number of different values per key dimension
        :param keys: the data keys to select from
        :param best_length: the length of the best-so-far set of runs,
            will be `-1` if no such best exists
        :return: the selected data
        """
        runs: list[dict[tuple[int, int, int], set[int]]] = [
            {} for _ in range(dims[KEY_INSTANCE])]
        n_combos: int = 0
        for key in keys:
            dct = runs[key[KEY_INSTANCE]]
            subkey = (key[KEY_ALGORITHM], key[KEY_ENCODING],
                      key[KEY_OBJECTIVE])
            if subkey in dct:
                dct[subkey].add(key[KEY_SEED])
            else:
                dct[subkey] = {key[KEY_SEED]}
                n_combos += 1

        if n_combos <= 0:
            raise ValueError("No runs??")

        inst_seeds: list[set[int]] = [
            next(iter(dct.values())) for dct in runs]
        min_set_len: int = tuple.__len__(keys)
        for i, dct in enumerate(runs):
            base = inst_seeds[i]
            for st in dct.values():
                base.intersection_update(st)
            sl: int = set.__len__(base)
            if sl < min_set_len:
                if sl <= 0:
                    logger("Cannot find non-empty instance/run intersection.")
                    return None
                if (sl * n_combos) <= best_length:
                    logger(f"Cannot surpass best length {best_length}.")
                    return None
                min_set_len = sl

        logger(f"Get {min_set_len} seeds per instance/algorithm/objective/"
               f"encoding combination.")
        use_seeds: list[list[int]] = [
            sorted(st)[:min_set_len] for st in inst_seeds]
        logger(f"Found {len(use_seeds)} seeds to use.")
        return [key for key in keys if key[KEY_SEED] in use_seeds[
            key[KEY_INSTANCE]]]

    def __str__(self) -> str:
        """
        Get the text representation of this selector.

        :returns "sameNumberOfRuns": always
        """
        return "sameNumberOfRuns"


def __seed_hash(x: tuple[str, int]) -> tuple[str, int]:
    """
    Compute a hash for instance-random seeds.

    The selection algorithm will deterministically choose which runs to
    keep. To make it truly deterministic, we always sort the settings
    according to the algorithm, instance, objective function, encoding, and
    random seed.

    When we create this order of configurations, we can normally rely on the
    natural or alphabetic order for algorithms, objectives, instances, and
    encodings.
    However, if we also use the natural order for random seeds, this might
    potentially lead to the problem that runs with larger random seeds are
    deleted more likely.
    Deleting some algorithms or instances is not really a problem, but using
    the seed value to select runs via a natural order of seeds could be
    problematic.
    The hash is based on a tuple containing the seed, which is different from
    the seed but still reliably the same in every run.
    This should lead to a uniform probability of seed deletion over the random
    seed spectrum in the absence of other criteria.

    :param x: the instance-random seeds
    :return: the hash
    """
    return x[0], hash((x[1], ))  # force hash to be different from x[1]


#: All combinations of instances/algorithms/encodings/objectives get the same
#: number of runs, and on all instances the same seeds are used.
#: This selector is quite fast and simple.
#: It will only delete runs, but attempt to retain all instance, algorithms,
#: objective, and encoding combinations.
#: It will thus fail if there are some combinations with mutually distinct
#: run sets.
SELECTOR_SAME_RUNS_FOR_ALL: Final[tuple[Selector, ...]] = (
    __SameNumberOfRuns(), )

#: A simple selector set for maximum runs.
#: This selector first attempts to do the selection given in
#: :const:`SELECTOR_SAME_RUNS_FOR_ALL` and then attempts to find
#: a larger set of consistent runs that might result from dropping off
#: instances, encodings, algorithms, or objectives.
SELECTOR_MAX_RUNS_SIMPLE: Final[tuple[Selector, ...]] = (
    *SELECTOR_SAME_RUNS_FOR_ALL, __HeuristicSelectorTiered())

#: Select the maximum number of runs in the most thoroughly possible way.
#: This selector performs many more different attempts compared to
#: :const:`SELECTOR_MAX_RUNS_SIMPLE`. It is very thorough in attempting to
#: find the largest possible set of consistent runs. It will therefore also
#: be much slower.
SELECTOR_MAX_RUNS: Final[tuple[Selector, ...]] = (
    *SELECTOR_MAX_RUNS_SIMPLE,
    __HeuristicSelectorNormTieredRev(),
    __HeuristicSelectorSum(), __HeuristicSelectorProduct(),
    __HeuristicSelectorMinimum(), __HeuristicSelectorNormTiered(),
    __HeuristicSelectorNormTieredRev(), __HeuristicSelectorNormSum())


def select_consistent(
        data: Iterable[T],
        selectors: Iterable[Selector] = SELECTOR_MAX_RUNS) -> list[T]:
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
    :param selectors: the selectors to use
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
    ['a1/i1/o1/e1/2', 'a1/i2/o1/e1/3', 'a2/i1/o1/e1/2', 'a2/i2/o1/e1/3']

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
    ['a1/i1/o1/e1/1', 'a1/i2/o1/e1/2']

    >>> list(map(__p, select_consistent((
    ...     a1i1o1e1s1, a2i1o1e1s2))))
    ['a1/i1/o1/e1/1']

    >>> list(map(__p, select_consistent((
    ...     a1i1o1e1s1, a2i1o1e1s2, a2i1o1e1s3))))
    ['a2/i1/o1/e1/2', 'a2/i1/o1/e1/3']

    >>> try:
    ...     select_consistent((a1i1o1e1s1, a1i1o1e1s2, a1i2o1e1s2, a1i2o1e1s2))
    ... except ValueError as ve:
    ...     print(str(ve)[:30])
    Found two records of type PerR

    >>> try:
    ...     select_consistent(1)
    ... except TypeError as te:
    ...     print(te)
    data should be an instance of typing.Iterable but is int, namely 1.

    >>> try:
    ...     select_consistent({234})
    ... except TypeError as te:
    ...     print(str(te)[:20])
    data[0] should be an

    >>> try:
    ...     select_consistent((a1i1o1e1s1, a1i1o1e1s2, a1i2o1e1s2), 1)
    ... except TypeError as te:
    ...     print(str(te)[:19])
    selectors should be

    >>> try:
    ...     select_consistent((a1i1o1e1s1, a1i1o1e1s2, a1i2o1e1s2), (1, ))
    ... except TypeError as te:
    ...     print(str(te)[:21])
    selectors[0] should b
    """
    if not isinstance(data, Iterable):
        raise type_error(data, "data", Iterable)
    if not isinstance(selectors, Iterable):
        raise type_error(selectors, "selectors", Iterable)

    # make data re-iterable
    use_data = data if isinstance(data, list | tuple) else list(data)
    total_len: Final[int] = len(use_data)
    if total_len <= 0:
        raise ValueError("The data is empty!")
    algorithm_set: set[str] = set()
    instance_set: set[str] = set()
    objective_set: set[str] = set()
    encoding_set: set[str] = set()
    seed_set: set[tuple[str, int]] = set()
    for i, item in enumerate(use_data):
        if not isinstance(item, PerRunData):
            raise type_error(item, f"data[{i}]", PerRunData)
        algorithm_set.add(item.algorithm)
        instance_set.add(item.instance)
        objective_set.add(item.objective)
        encoding_set.add(item.encoding)
        seed_set.add((item.instance, item.rand_seed))

    algorithms: Final[list[str]] = sorted(algorithm_set)
    del algorithm_set
    instances: Final[list[str]] = sorted(instance_set)
    del instance_set
    objectives: Final[list[str]] = sorted(objective_set)
    del objective_set
    encodings: Final[list[str | None]] = sorted(
        encoding_set, key=lambda s: "" if s is None else s)
    del encoding_set
    # Random seeds
    seeds: Final[list[tuple[str, int]]] = sorted(seed_set, key=__seed_hash)
    del seed_set
    dims: Final[tuple[int, int, int, int, int]] = (
        list.__len__(algorithms), list.__len__(instances),
        list.__len__(objectives), list.__len__(encodings),
        list.__len__(seeds))

    logger(f"Found {dims[0]} algorithms, {dims[1]} instances, "
           f"{dims[2]} objectives, {dims[3]} encodings, and "
           f"{dims[4]} instance/seed combinations over "
           f"{total_len} runs in total.")

    data_map: dict[tuple[int, int, int, int, int], T] = {}
    for i, item in enumerate(use_data):
        key = (algorithms.index(item.algorithm),
               instances.index(item.instance),
               objectives.index(item.objective),
               encodings.index(item.encoding),
               seeds.index((item.instance, item.rand_seed)))
        if key in data_map:
            raise ValueError(f"Found two records of type {item},"
                             f"the second one at index {i}.")
        data_map[key] = item

    source: tuple[tuple[int, int, int, int, int], ...] = tuple(sorted(
        data_map.keys()))
    best_data: list | None = None
    best_length: int = -1

    # Apply all the selectors one by one.
    for i, selector in enumerate(selectors):
        if not isinstance(selector, Selector):
            raise type_error(selector, f"selectors[{i}]", Selector)
        selector_name = str(selector)
        logger(f"Now invoking selector {selector_name!r}.")
        best_data_2: list | None = selector.select(
            dims, source, best_length)
        if best_data_2 is not None:
            bdl2 = list.__len__(best_data_2)
            if bdl2 > best_length:
                logger(f"Selector {selector_name!r} found new best "
                       f"of {bdl2} runs.")
                best_length = bdl2
                best_data = best_data_2
                if best_length >= total_len:
                    logger("All runs can be retained, we can stop here.")
                    best_data.clear()
                    best_data.extend(use_data)
                    best_data.sort()
                    return best_data

        del best_data_2

    if best_length <= 0:
        raise ValueError("Did not find consistent run subset.")
    logger(f"After applying all selectors, we got {best_length} records.")

    # Now we construct the final result.
    best_data.sort()
    for i, key in enumerate(best_data):
        best_data[i] = data_map.pop(key)
    del data_map
    best_data.sort()
    return best_data
