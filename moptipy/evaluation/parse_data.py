"""A module for parsing the different section types of log files."""
from typing import Dict, Iterable

import moptipy.utils.logging as logging


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
    dct = {}
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
