"""A module for parsing the different section types of log files."""
from typing import Dict, Iterable, List, Callable

import numpy as np

import moptipy.utils.logging as logging
from moptipy.utils.cache import is_new
from moptipy.utils.nputils import strs_to_ints, strs_to_uints,\
    strs_to_bools, strs_to_floats


def parse_key_values(lines: Iterable[str]) -> Dict[str, str]:
    """
    Parse a :meth:`~moptipy.utils.logger.Logger.key_values` section's text.

    :param Iterable[str] lines: the lines with the key-values pairs
    :return: the dictionary with the
    :rtype: Dict[str, str]

    >>> from moptipy.utils.logger import InMemoryLogger
    >>> with InMemoryLogger() as l:
    ...     with l.key_values("B") as kv:
    ...         kv.key_value("a", "b")
    ...         with kv.scope("c") as kvc:
    ...             kvc.key_value("d", 12)
    ...             kvc.key_value("e", True)
    ...         kv.key_value("f", 3)
    ...     txt = l.get_log()
    >>> print(txt)
    ['BEGIN_B', 'a: b', 'c.d: 12', 'c.e: T', 'f: 3', 'END_B']
    >>> dic = parse_key_values(txt[1:5])
    >>> keys = list(dic.keys())
    >>> keys.sort()
    >>> print(keys)
    ['a', 'c.d', 'c.e', 'f']
    """
    if not isinstance(lines, Iterable):
        raise TypeError(
            f"lines must be Iterable of strings, but is {type(lines)}.")
    dct = dict()
    for line in lines:
        splt = line.split(logging.KEY_VALUE_SEPARATOR)
        if len(splt) != 2:
            raise ValueError(
                f"Two strings separated by '{logging.KEY_VALUE_SEPARATOR}' "
                f"expected, but encountered {len(splt)} in '{line}'.")
        key = splt[0].strip()
        if len(key) <= 0:
            raise ValueError(f"Empty key encountered in '{line}'.")
        value = splt[1].strip()
        if len(value) <= 0:
            raise ValueError(f"Empty value encountered in '{line}'.")
        dct[key] = value

    return dct


def strs_to_array(rows: Iterable[str]) -> np.ndarray:
    """
    Convert an iterable of strings to a list of one of the permitted types.

    In a logging CSV section, we permit `bool`, `int`, and `float` arguments.
    This function can translate lists of the string
    representations of these arguments to lists of these primitve types.

    :param Iterable[str] rows: the rows to convert
    :return: the list
    :rtype: np.array
    :raises ValueError: if the list cannot be converted

    >>> strs_to_array(["1", "2", "3"])
    array([1, 2, 3], dtype=uint64)
    >>> strs_to_array(["1", "-2", "3"])
    array([ 1, -2,  3])
    >>> strs_to_array(["1", "2.2", "3"])
    array([1. , 2.2, 3. ])
    >>> strs_to_array(["T", "F", "T"])
    array([ True, False,  True])
    """
    can_be_bool = True
    can_be_int = True
    can_be_float = True
    is_signed = False

    for s in rows:
        if s in ("F", "T"):
            can_be_float = False
            can_be_int = False
        else:
            can_be_bool = False
            if s.startswith("-"):
                is_signed = True
            if ("e" in s) or ("." in s) or ("E" in s):
                can_be_int = False

    if can_be_bool:
        return strs_to_bools(rows)
    if can_be_int:
        if is_signed:
            return strs_to_ints(rows)
        return strs_to_uints(rows)
    if can_be_float:
        return strs_to_floats(rows)
    raise ValueError("Data is invalid.")


def parse_csv(lines: List[str],
              default: Callable = strs_to_array,
              **kwargs) -> Dict[str, np.ndarray]:
    """
    Parse lines of CSV data and obtain a dictionary of numpy ndarrays.

    :param List[str] lines: the lines
    :param Callable default: the default parser for any column not in kwargs
    :return: a dictionary of :class:`np.ndarray`
    :rtype: Dict[str, np.ndarray]

    >>> res = parse_csv(["a;b;c;d", "1;T;-3;0", "0.5;F;4;5", "1.5;T;123;7"])
    >>> res["a"]
    array([1. , 0.5, 1.5])
    >>> res["b"]
    array([ True, False,  True])
    >>> res["c"]
    array([ -3,   4, 123])
    >>> res["c"].dtype
    dtype('int64')
    >>> res["d"]
    array([0, 5, 7], dtype=uint64)
    """
    if not isinstance(lines, list):
        raise TypeError(
            f"lines must be list of strings, but is {type(lines)}.")

    n_rows = len(lines)
    if n_rows < 2:
        raise ValueError("lines must contain at least two elements, but "
                         f"contains {n_rows}.")

    columns = [c.strip() for c in lines[0].split(logging.CSV_SEPARATOR)]
    n_cols = len(columns)
    if n_cols < 1:
        raise ValueError("There must be at least one column, but found none "
                         f"in '{lines[0]}'.")
    cache = is_new()
    for col in columns:
        if len(col) <= 0:
            raise ValueError(
                f"Encountered empty column name in '{lines[0]}'.")
        if not cache(col):
            raise ValueError(f"Column '{col}' appears twice.")
    del cache

    matrix = list(zip(*[[c.strip() for c in line.split(logging.CSV_SEPARATOR)]
                        for line in lines[1:]]))
    n_rows = n_rows - 1
    if n_cols != len(matrix):
        raise ValueError(f"Number of expected columns: {n_cols}, "
                         f"number columns found: {len(matrix)}.")

    result = {}
    for i, col in enumerate(columns):
        parser = kwargs[col] if col in kwargs else default
        if parser is None:
            continue
        cc = parser(matrix[i])
        if len(cc) != n_rows:
            raise ValueError("Obtained incorrect length of row, "
                             f"should be {n_rows}, but is {len(cc)}.")
        result[col] = cc

    return result
