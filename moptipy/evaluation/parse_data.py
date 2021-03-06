"""A module for parsing the different section types of log files."""
from typing import Dict, Iterable

import yaml


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
    ['BEGIN_B', 'a: b', 'c.d: 12', 'c.e: True', 'f: 3', 'END_B']
    >>> dic = parse_key_values(txt[1:5])
    >>> print(list(dic.keys()))
    ['a', 'c.d', 'c.e', 'f']
    """
    if not isinstance(lines, Iterable):
        raise TypeError("lines must be Iterable of strings, but is "
                        + str(type(lines)) + ".")
    text = "\n".join(lines)
    if len(text) <= 0:
        return {}
    return yaml.safe_load(text)
