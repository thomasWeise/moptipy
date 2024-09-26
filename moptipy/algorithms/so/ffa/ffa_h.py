"""Here we provide some tools for implementing FFA."""

from collections import Counter
from io import StringIO
from math import nan
from typing import Callable, Final

import numpy as np
from pycommons.io.csv import CSV_SEPARATOR
from pycommons.math.int_math import try_int
from pycommons.types import type_error

from moptipy.api.objective import Objective
from moptipy.api.process import Process
from moptipy.utils.nputils import DEFAULT_INT

#: the log section for the frequency table
H_LOG_SECTION: Final[str] = "H"

#: The difference between upper- and lower bound at which we switch from
#: using arrays as backing store for the frequency table H to maps.
SWITCH_TO_MAP_RANGE: Final[int] = 67_108_864


def create_h(objective: Objective) -> tuple[
        np.ndarray | Counter[int | float], int]:
    """
    Create a data structure for recording frequencies.

    If our objective function always returns integer values and its range is
    not too large, then we can use a :class:`numpy.ndarray` as a backing
    datastructure for the `H`-map used in FFA. If the objective function may
    sometimes return `float` values or has a wide range, it will be better
    (or even necessary) to use :class:`Counter` instead.

    This function here makes the decision of what backing datastructure to
    use.

    Additionally, it may be the lower or even upper bound of an integer
    objective function could be negative. To avoid dealing with negative
    indices, we would then add an offset to each objective value before
    accessing it. This function will return an appropriate offset as well.

    :param objective: the objective for which the data structure should be
        created
    :return: a tuple of the `H`-data structure as well as an offset to be
        added to all objective values.
    """
    if not isinstance(objective, Objective):
        raise type_error(objective, "objective", Objective)
    if objective.is_always_integer():
        lb: Final[int | float] = objective.lower_bound()
        ub: Final[int | float] = objective.upper_bound()
        if isinstance(ub, int) and isinstance(lb, int) \
                and ((ub - lb) <= SWITCH_TO_MAP_RANGE):
            return np.zeros(ub - lb + 1, DEFAULT_INT), -lb
    return Counter(), 0


def clear_h(h: np.ndarray | Counter[int | float]) -> None:
    """
    Clear a H-Table.

    :param h: the H-table to clear.
    """
    if isinstance(h, np.ndarray):
        h.fill(0)
    elif isinstance(h, dict):
        h.clear()
    else:
        raise type_error(h, "h", (np.ndarray, Counter))


def h_to_str(
        h: np.ndarray | Counter[int | float],
        offset: int) -> str:
    """
    Convert a `H`-table to a string.

    :param h: the `H`-table, as created by :func:`create_h`.
    :param offset: the offset, as computed by :func:`create_h`.

    >>> hl = np.array([0, 0, 1, 7, 4, 0, 0, 9, 0])
    >>> h_to_str(hl, 0)
    '2;1;;7;;4;7;9'
    >>> h_to_str(hl, -1)
    '3;1;;7;;4;8;9'
    >>> hd = Counter((1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3,
    ...               3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2))
    >>> h_to_str(hd, 0)
    '1;5;;9;;6;;7'
    >>> try:
    ...     hd = {1: 0}
    ...     h_to_str(hd, 0)
    ... except ValueError as ve:
    ...     print(ve)
    empty H table?
    >>> hx = np.zeros(100, int)
    >>> hx[10] = 4
    >>> hx[12] = 234
    >>> hx[89] = 111
    >>> hx[90] = 1
    >>> hx[45] = 2314
    >>> h_to_str(hx, 0)
    '10;4;12;234;45;2314;89;111;;1'
    """
    if not isinstance(h, np.ndarray | dict):
        raise type_error(h, "h", (np.ndarray, Counter))
    if not isinstance(offset, int):
        raise type_error(offset, "offset", int)
    csep: Final[str] = CSV_SEPARATOR
    sep: str = ""

    with StringIO() as out:
        write: Callable[[str], int] = out.write  # fast call
        old_index: int | float = nan

        # We iterate over the whole `H` table.
        # If the table is an np.array, then we just use the indices to
        # iterate directly. If it is a `Counter`, then we iterate over
        # its keys. If there are not too many keys, then we sort them
        # first.
        for i in range(len(h)) if isinstance(h, np.ndarray) else (sorted(
                h.keys()) if len(h) <= SWITCH_TO_MAP_RANGE else h.keys()):
            value: int | float = h[i]  # type: ignore
            if value <= 0:  # skip over values that have never been hit
                continue
            write(sep)  # write separator (nothing upon first call)
            sep = csep  # now the next call will write ;
            use_index: int | float = i - offset  # subtract the offset
            if isinstance(use_index, float):  # if it's float, try to convert
                use_index = try_int(use_index)
            if (use_index - 1) != old_index:  # we skip if current = old + 1
                write(str(use_index))
            old_index = use_index  # step the index
            write(csep)  # write separator to frequency counter
            write(str(value))  # write the frequency
        res: Final[str] = out.getvalue()  # get the final string

    if str.__len__(res) <= 0:
        raise ValueError("empty H table?")
    return res


def log_h(process: Process, h: np.ndarray | Counter[int | float],
          offset: int) -> None:
    """
    Convert a frequency table `H` to a string and log it to a process.

    The frequency table is logged as a single line of text into a section
    `H` delimited by the lines `BEGIN_H` and `END_H`. The line consists
    of `2*n` semicolon separated values. Each such value pair consists of
    an objective value `y` and its observed frequency `H[y]`. The former is
    either an integer or a float and the latter is an integer.
    If two consecutive objective values take the form 'f` and `f+1`, then
    the second one will be omitted, and so on.

    :param process: the process
    :param h: the `H`-table, as created by :func:`create_h`.
    :param offset: the offset, as computed by :func:`create_h`.
    """
    if not isinstance(process, Process):
        raise type_error(process, "process", Process)
    if process.has_log():
        process.add_log_section(H_LOG_SECTION, h_to_str(h, offset))
