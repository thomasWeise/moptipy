"""Some basic type handling routines."""
from typing import Any, Type, Iterable, Union, Final


def type_name(tpe: Type) -> str:
    """
    Convert a type to a string.

    :param tpe: the type
    :returns: the string

    >>> type_name(type(None))
    'None'
    >>> type_name(int)
    'int'
    >>> from moptipy.utils.path import Path
    >>> type_name(Path)
    'moptipy.utils.path.Path'
    """
    c1: str = str(tpe)
    if c1.startswith("<class '"):
        c1 = c1[8:-2]
    if c1 == "NoneType":
        return "None"

    if hasattr(tpe, "__qualname__"):
        c2: str = tpe.__qualname__
        if hasattr(tpe, "__module__"):
            module = tpe.__module__
            if (module is not None) and (module != "builtins"):
                c2 = f"{module}.{c2}"
        if len(c2) > len(c1):
            return c2
    return c1


def type_name_of(obj) -> str:
    """
    Get the fully-qualified class name of an object.

    :param obj: the object
    :returns: the fully-qualified class name of the object

    >>> from moptipy.examples.jssp.instance import Instance
    >>> type_name_of(Instance.from_resource("demo"))
    'moptipy.examples.jssp.instance.Instance'
    >>> from numpy.random import default_rng
    >>> type_name_of(default_rng())
    'numpy.random._generator.Generator'
    """
    if obj is None:
        return "None"
    c1: Final[str] = type_name(type(obj))
    if hasattr(obj, "__class__"):
        cls: Final[Type] = obj.__class__
        c2: str = type_name(cls)

        if hasattr(cls, "__qualname__"):
            c3: str = cls.__qualname__
            if hasattr(obj, "__module__"):
                module = obj.__module__
                if (module is not None) and (module != "builtins"):
                    c3 = f"{module}.{c3}"
            if len(c3) > len(c2):
                c2 = c3

        if len(c2) > len(c1):
            return c2
    return c1


def type_error(obj: Any,
               name: str,
               expected: Union[None, Type, Iterable[Type]] = None,
               call: bool = False) -> Union[ValueError, TypeError]:
    """
    Create an error to raise if a type did not fit.

    :param obj: the object that is of the wrong type
    :param name: the name of the object
    :param expected: the expected types (or `None`)
    :param call: the object should have been callable?
    :returns: a :class:`TypeError` with a descriptive information

    >>> type_error(1.3, "var", int)
    TypeError("var should be an instance of int but is float, namely '1.3'.")
    >>> type_error("x", "z", (int, float)).args[0]
    "z should be an instance of any in {float, int} but is str, namely 'x'."
    >>> type_error("f", "q", call=True).args[0]
    "q should be a callable but is str, namely 'f'."
    >>> type_error("1", "2", bool, call=True).args[0]
    "2 should be an instance of bool or a callable but is str, namely '1'."
    >>> type_error(None, "x", str)
    TypeError('x should be an instance of str but is None.')
    """
    exp: str = ""
    if isinstance(expected, Iterable):
        exp = ', '.join(sorted([type_name(e) for e in expected]))
        exp = f"an instance of any in {{{exp}}}"
    elif expected is not None:
        exp = f"an instance of {type_name(expected)}"
    if call:
        exp = f"{exp} or a callable" if exp else "a callable"

    if obj is None:
        return TypeError(f"{name} should be {exp} but is None.")
    return TypeError(f"{name} should be {exp} but is "
                     f"{type_name_of(obj)}, namely '{obj}'.")
