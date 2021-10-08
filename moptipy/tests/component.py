"""Functions that can be used to test component implementations."""
from moptipy.api.component import Component
from moptipy.utils import logging
from moptipy.utils.logger import InMemoryLogger


def test_component(component: Component) -> None:
    """
    Check whether an object is a moptipy component.

    :param component: the component to test
    :raises ValueError: if `component` is not a valid Component
    """
    if not isinstance(component, Component):
        raise ValueError("Expected to receive an instance of Component, but "
                         f"got a '{type(component)}'.")

    name = component.get_name()
    if not isinstance(name, str):
        raise ValueError(
            f"get_name() must return a string, but returns a '{type(name)}'.")

    clean_name = logging.sanitize_name(name)
    if clean_name != name:
        raise ValueError(
            "get_name() must return a string which does not change when being"
            f" sanitized, but returned '{name}',"
            f" which becomes '{clean_name}'.")

    strstr = str(component)
    if strstr != name:
        raise ValueError(
            f"str(component) must equal get_name(), but got '{strstr}' "
            f"vs. '{name}'.")

    strstr = repr(component)
    if strstr != name:
        raise ValueError("repr(component) must equal get_name(), but got '"
                         f"{strstr}' vs. '{name}'.")

    with InMemoryLogger() as log:
        with log.key_values("KV") as kv:
            component.log_parameters_to(kv)
        lines = log.get_log()

    ll = len(lines) - 1
    if (lines[0] != logging.SECTION_START + "KV") or \
            (lines[ll] != logging.SECTION_END + "KV"):
        raise ValueError("Invalid log data produced '"
                         + "\n".join(lines) + "'.")

    lines = lines[1:ll]
    if len(lines) < 2:
        raise ValueError("A component must produce at least two lines of "
                         f"Key-Value data, but produced {len(lines)}.")

    if not lines[0].startswith(logging.KEY_NAME):
        raise ValueError(
            f"First log line must begin with '{logging.KEY_NAME}', but starts"
            f" with '{lines[0]}'.")

    if not lines[1].startswith(logging.KEY_TYPE):
        raise ValueError(
            f"Second log line must begin with '{logging.KEY_TYPE}', but "
            f"starts with '{lines[1]}'.")
