"""A factory for functions checking whether argument values are new."""
from typing import Any, Callable, Dict


def is_new() -> Callable[[str], bool]:
    """
    Create a function returning `True` when seeing new values.

    Creates a function which returns `True` only the first time it receives a
    given argument and `False` all subsequent times.
    This is based on https://stackoverflow.com/questions/27427067

    :returns: a function `is_new(xx)` that will return `True` the first time
        it encounters any value `xx` and `False` for all values it has already
        seen

    >>> from moptipy.utils.cache import is_new
    >>> check = is_new()
    >>> print(check("a"))
    True
    >>> print(check("a"))
    False
    >>> print(check("b"))
    True
    >>> print(check("b"))
    False
    """
    s: Dict[Any, int] = {}
    setdefault = s.setdefault
    n = 0

    def add(x):
        nonlocal n
        n += 1
        return setdefault(x, n) == n

    return add
