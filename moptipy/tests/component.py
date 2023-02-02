"""Functions that can be used to test component implementations."""
from typing import Final

from moptipy.api import logging
from moptipy.api.component import Component
from moptipy.utils.logger import (
    KEY_VALUE_SEPARATOR,
    SECTION_END,
    SECTION_START,
    InMemoryLogger,
)
from moptipy.utils.strings import sanitize_name
from moptipy.utils.types import type_error, type_name_of


def validate_component(component: Component) -> None:
    """
    Check whether an object is a valid moptipy component.

    This test checks the conversion to string and the logging of parameters.
    This method must be called before the other `validate_*` methods provided
    in this package.

    :param component: the component to test
    :raises ValueError: if `component` is not a valid
        :class:`~moptipy.api.component.Component` instance
    :raises TypeError: if a type is wrong or `component` is not even an
        instance of :class:`~moptipy.api.component.Component`
    """
    if not isinstance(component, Component):
        raise type_error(component, "component", Component)
    if component.__class__ == Component:
        raise ValueError(
            "component cannot be an instance of Component directly.")

    name = str(component)
    if not isinstance(name, str):
        raise type_error(name, "str(component)", str)
    if len(name) <= 0:
        raise ValueError("str(component) must return a non-empty string, "
                         f"but returns a {name!r}.")
    if name.strip() != name:
        raise ValueError("str(component) must return a string without "
                         "leading or trailing white space, "
                         f"but returns a {name!r}.")

    clean_name = sanitize_name(name)
    if clean_name != name:
        raise ValueError(
            "str(component) must return a string which does not "
            f"change when being sanitized, but returned {name!r},"
            f" which becomes {clean_name!r}.")
    name = str(component)
    if clean_name != name:
        raise ValueError("str(component) must always return the same value, "
                         f"but returns a {name!r} and {clean_name!r}.")

    name = repr(component)
    if name != clean_name:
        raise ValueError("repr(component) must equal str(component), but "
                         f"got {clean_name!r} vs. {name!r}.")

    if not (hasattr(component, "initialize")
            and callable(getattr(component, "initialize"))):
        raise ValueError("component must have method initialize.")
    component.initialize()

    if name != str(component):
        raise ValueError(f"name changed to {str(component)!r} "
                         f"from {name!r} after initialize!")

    # test the logging of parameter values
    if not (hasattr(component, "log_parameters_to")
            and callable(getattr(component, "log_parameters_to"))):
        raise ValueError("component must have method log_parameters_to.")

    secname: Final[str] = "KV"
    with InMemoryLogger() as log:
        with log.key_values(secname) as kv:
            component.log_parameters_to(kv)
        lines = log.get_log()

    ll = len(lines) - 1
    if (lines[0] != SECTION_START + secname) or \
            (lines[ll] != SECTION_END + secname):
        raise ValueError("Invalid log data produced '"
                         + "\n".join(lines) + "'.")

    kvs: Final[str] = KEY_VALUE_SEPARATOR
    lines = lines[1:ll]
    if len(lines) < 2:
        raise ValueError("A component must produce at least two lines of "
                         f"key-value data, but produced {len(lines)}.")

    done_keys: set[str] = set()
    idx: int = 0
    key: str = logging.KEY_NAME
    done_keys.add(key)
    keystr: str = f"{key}{kvs}"
    line: str = lines[idx]
    idx += 1
    if not line.startswith(keystr):
        raise ValueError(
            f"First log line must begin with {keystr!r}, but starts"
            f" with {line!r}.")
    rest = line[len(keystr):]
    if rest != name:
        raise ValueError(
            f"value of key {keystr!r} should equal "
            f"{name!r} but is {rest!r}.")

    key = logging.KEY_CLASS
    keystr = f"{key}{kvs}"
    done_keys.add(key)
    line = lines[idx]
    idx += 1
    if not line.startswith(keystr):
        raise ValueError(
            f"Second log line must begin with {keystr!r}, but "
            f"starts with {line!r}.")
    rest = line[len(keystr):]
    want = type_name_of(component)
    if rest != want:
        raise ValueError(
            f"value of key {keystr!r} should equal "
            f"{want!r} but is {rest!r}.")

    for line in lines[idx:]:
        i = line.index(kvs)
        key = line[0:i].strip()
        b = line[i + len(kvs)].strip()
        if (len(key) <= 0) or (len(b) <= 0):
            raise ValueError(
                f"Invalid key-value pair {line!r} - "
                f"splits to {(key + kvs + b)!r}!")
        if key in done_keys:
            raise ValueError(f"key {key!r} appears twice!")
        done_keys.add(key)
