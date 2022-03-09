"""Functions that can be used to test component implementations."""
from typing import Final

from moptipy.api import logging
from moptipy.api.component import Component
from moptipy.utils.logger import InMemoryLogger
from moptipy.utils.types import classname


def test_component(component: Component) -> None:
    """
    Check whether an object is a valid moptipy component.

    This test checks the conversion to string and the logging of parameters.

    :param component: the component to test
    :raises ValueError: if `component` is not a valid Component
    """
    if not isinstance(component, Component):
        raise ValueError("Expected to receive an instance of Component, but "
                         f"got a '{type(component)}'.")

    name = str(component)
    if not isinstance(name, str):
        raise ValueError(
            f"get_name() must return a string, but returns a '{type(name)}'.")

    clean_name = logging.sanitize_name(name)
    if clean_name != name:
        raise ValueError(
            "get_name() must return a string which does not change when being"
            f" sanitized, but returned '{name}',"
            f" which becomes '{clean_name}'.")

    strstr = repr(component)
    if strstr != name:
        raise ValueError("repr(component) must equal __str__(), but got '"
                         f"{strstr}' vs. '{name}'.")

    # test the logging of parameter values
    secname: Final[str] = "KV"
    with InMemoryLogger() as log:
        with log.key_values(secname) as kv:
            component.log_parameters_to(kv)
        lines = log.get_log()

    ll = len(lines) - 1
    if (lines[0] != logging.SECTION_START + secname) or \
            (lines[ll] != logging.SECTION_END + secname):
        raise ValueError("Invalid log data produced '"
                         + "\n".join(lines) + "'.")

    kvs: Final[str] = logging.KEY_VALUE_SEPARATOR
    lines = lines[1:ll]
    if len(lines) < 2:
        raise ValueError("A component must produce at least two lines of "
                         f"Key-Value data, but produced {len(lines)}.")

    keystr: str = f"{logging.KEY_NAME}{kvs}"
    line: str = lines[0]
    if not line.startswith(keystr):
        raise ValueError(
            f"First log line must begin with '{keystr}', but starts"
            f" with '{line}'.")
    rest = line[len(keystr):]
    if rest != name:
        raise ValueError(
            f"value of key '{keystr}' should equal "
            f"'{name}' but is '{rest}'.")

    keystr = f"{logging.KEY_CLASS}{kvs}"
    line = lines[1]
    if not line.startswith(logging.KEY_CLASS):
        raise ValueError(
            f"Second log line must begin with '{keystr}', but "
            f"starts with '{line}'.")
    rest = line[len(keystr):]
    want = classname(component)
    if rest != want:
        raise ValueError(
            f"value of key '{keystr}' should equal "
            f"'{want}' but is '{rest}'.")

    for line in lines[2:]:
        i = line.index(kvs)
        a = line[0:i].strip()
        b = line[i + len(kvs)].strip()
        if (len(a) <= 0) or (len(b) <= 0):
            raise ValueError(
                f"Invalid key-value pair '{line}' - splits to '{a}: {b}'!")
