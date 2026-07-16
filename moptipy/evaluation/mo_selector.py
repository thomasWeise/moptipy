"""
A tool for selecting multi-objective data.

Different from :func:`~moptipy.evaluation.selector.select_consistent`, the
function :func:`~moptipy.evaluation.mo_selector.mo_select_consistent` can
deal with multi-objective data.
The difference is that such data may contain multiple results per run.
"""


from operator import itemgetter
from typing import Iterable, TypeVar

from pycommons.types import type_error

from moptipy.evaluation.base import PerRunData
from moptipy.evaluation.selector import (
    SELECTOR_MAX_RUNS,
    Selector,
    select_consistent,
)

#: the type variable for the selector routine
T = TypeVar("T", bound=PerRunData)


def mo_select_consistent(
        data: Iterable[T],
        selectors: Iterable[Selector] = SELECTOR_MAX_RUNS) -> list[T]:
    """
    Select a consistent subset of data which may be multi-objective.

    :param data: The input data stream
    :param selectors: the selectors that should be used
    :return: the selected data

    >>> def __p(x) -> str:
    ...     return (f"{x.algorithm}/{x.instance}/{x.objective}/{x.encoding}/"
    ...             f"{x.rand_seed}")

    >>> from moptipy.evaluation.selector import SELECTOR_SAME_SEEDS_FOR_ALL

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

    >>> list(map(__p, mo_select_consistent((
    ...     a1i1o1e1s1, a1i1o1e1s2, a1i1o1e1s3, a1i2o1e1s2,
    ...     a1i2o1e1s1, a1i2o1e1s2, a1i2o1e1s3,
    ...     a2i1o1e1s1, a2i1o1e1s2, a1i1o1e1s1,
    ...     a2i2o1e1s2, a2i2o1e1s3))))
    ['a1/i1/o1/e1/1', 'a1/i1/o1/e1/1', 'a1/i1/o1/e1/2', 'a1/i2/o1/e1/2', \
'a1/i2/o1/e1/2', 'a1/i2/o1/e1/3', 'a2/i1/o1/e1/1', 'a2/i1/o1/e1/2', \
'a2/i2/o1/e1/2', 'a2/i2/o1/e1/3']

    >>> list(map(__p, mo_select_consistent((
    ...     a1i1o1e1s2, a1i1o1e1s3, a1i1o1e1s3, a1i2o1e1s1,
    ...     a1i2o1e1s1, a1i2o1e1s2, a1i2o1e1s3, a1i2o1e1s1,
    ...     a2i1o1e1s1, a2i1o1e1s2,
    ...     a2i2o1e1s2, a2i2o1e1s3))))
    ['a1/i1/o1/e1/2', 'a1/i2/o1/e1/3', 'a2/i1/o1/e1/2', 'a2/i2/o1/e1/3']

    >>> list(map(__p, mo_select_consistent((
    ...     a1i1o1e1s2, a1i1o1e1s3, a1i1o1e1s3, a1i1o1e1s3,
    ...     a1i2o1e1s1, a1i2o1e1s2, a1i2o1e1s3,
    ...     a2i1o1e1s1, a2i1o1e1s2, a1i1o1e1s3,
    ...     a2i2o1e1s2))))
    ['a1/i1/o1/e1/2', 'a1/i2/o1/e1/2', 'a2/i1/o1/e1/2', 'a2/i2/o1/e1/2']

    >>> list(map(__p, mo_select_consistent((
    ...     a1i1o1e1s1, a1i1o1e1s2, a1i1o1e1s3, a1i1o1e1s2, a1i1o1e1s2,
    ...     a2i2o1e1s1, a2i2o1e1s2, a2i2o1e1s3))))
    ['a1/i1/o1/e1/1', 'a1/i1/o1/e1/2', 'a1/i1/o1/e1/2', 'a1/i1/o1/e1/2', \
'a1/i1/o1/e1/3']

    >>> list(map(__p, mo_select_consistent((
    ...     a1i1o1e1s1, a1i1o1e1s2, a1i1o1e1s3, a2i1o1e1s2, a2i1o1e1s2,
    ...     a2i1o1e1s1, a2i1o1e1s2, a2i1o1e1s3))))
    ['a1/i1/o1/e1/1', 'a1/i1/o1/e1/2', 'a1/i1/o1/e1/3', 'a2/i1/o1/e1/1', \
'a2/i1/o1/e1/2', 'a2/i1/o1/e1/2', 'a2/i1/o1/e1/2', 'a2/i1/o1/e1/3']

    >>> list(map(__p, mo_select_consistent((
    ...     a1i1o1e1s1, a1i1o1e1s1, a1i1o1e1s2, a1i2o1e1s2, a1i2o1e1s3))))
    ['a1/i1/o1/e1/1', 'a1/i1/o1/e1/1', 'a1/i1/o1/e1/2', 'a1/i2/o1/e1/2', \
'a1/i2/o1/e1/3']

    >>> list(map(__p, mo_select_consistent((
    ...     a1i1o1e1s1, a1i1o1e1s2, a1i1o1e1s2, a1i2o1e1s2))))
    ['a1/i1/o1/e1/1', 'a1/i2/o1/e1/2']

    >>> list(map(__p, mo_select_consistent((
    ...     a1i1o1e1s1, a1i1o1e1s1, a1i1o1e1s1, a1i1o1e1s1, a2i1o1e1s2))))
    ['a1/i1/o1/e1/1', 'a1/i1/o1/e1/1', 'a1/i1/o1/e1/1', 'a1/i1/o1/e1/1']

    >>> list(map(__p, mo_select_consistent((
    ...     a1i1o1e1s1, a2i1o1e1s2, a2i1o1e1s3))))
    ['a2/i1/o1/e1/2', 'a2/i1/o1/e1/3']

    >>> list(map(__p, mo_select_consistent((
    ...     a1i1o1e1s1, a1i1o1e1s2, a1i1o1e1s3,
    ...     a1i2o1e1s1, a1i2o1e1s2, a1i2o1e1s3,
    ...     a2i1o1e1s1, a2i1o1e1s2,
    ...     a2i2o1e1s2, a2i2o1e1s3), SELECTOR_SAME_SEEDS_FOR_ALL)))
    ['a1/i1/o1/e1/1', 'a1/i1/o1/e1/2', 'a1/i2/o1/e1/2', 'a1/i2/o1/e1/3', \
'a2/i1/o1/e1/1', 'a2/i1/o1/e1/2', 'a2/i2/o1/e1/2', 'a2/i2/o1/e1/3']

    >>> list(map(__p, mo_select_consistent((
    ...     a1i1o1e1s1, a1i1o1e1s2, a1i1o1e1s3,
    ...     a1i2o1e1s1, a1i2o1e1s2, a1i2o1e1s3,
    ...     a2i1o1e1s1, a2i1o1e1s2,
    ...     a2i2o1e1s2, ), SELECTOR_SAME_SEEDS_FOR_ALL)))
    ['a1/i1/o1/e1/1', 'a1/i1/o1/e1/2', 'a1/i2/o1/e1/2', \
'a2/i1/o1/e1/1', 'a2/i1/o1/e1/2', 'a2/i2/o1/e1/2']

    >>> try:
    ...     mo_select_consistent(1)
    ... except TypeError as te:
    ...     print(te)
    data should be an instance of typing.Iterable but is int, namely 1.

    >>> try:
    ...     mo_select_consistent({234})
    ... except TypeError as te:
    ...     print(str(te)[:20])
    data[0] should be an

    >>> try:
    ...     mo_select_consistent((a1i1o1e1s1, a1i1o1e1s2, a1i2o1e1s2), 1)
    ... except TypeError as te:
    ...     print(str(te)[:19])
    selectors should be

    >>> try:
    ...     mo_select_consistent((a1i1o1e1s1, a1i1o1e1s2, a1i2o1e1s2), (1, ))
    ... except TypeError as te:
    ...     print(str(te)[:21])
    selectors[0] should b
    """
    if not isinstance(data, Iterable):
        raise type_error(data, "data", Iterable)
    if not isinstance(selectors, Iterable):
        raise type_error(selectors, "selectors", Iterable)

    mo_map: dict[tuple[str, str, str, str | None, int], list[T]] = {}
    for i, item in enumerate(data):
        if not isinstance(item, PerRunData):
            raise type_error(item, f"data[{i}]", PerRunData)
        key = (item.instance, item.algorithm, item.objective, item.encoding,
               item.rand_seed)
        if key in mo_map:
            mo_map[key].append(item)
        else:
            mo_map[key] = [item]

    raw: list[T] = select_consistent(map(
        itemgetter(0), mo_map.values()), selectors)
    result: list[T] = []
    for item in raw:
        key = (item.instance, item.algorithm, item.objective, item.encoding,
               item.rand_seed)
        result.extend(mo_map[key])
    result.sort()
    return result
