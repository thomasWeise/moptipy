"""Functions that can be used to test spaces."""
from typing import Any, Callable

# noinspection PyPackageRequirements
from pytest import raises

from moptipy.api.space import Space, check_space
from moptipy.tests.component import validate_component
from moptipy.utils.logger import COMMENT_CHAR, SECTION_END, SECTION_START
from moptipy.utils.types import type_error


def validate_space(
        space: Space,
        make_element_valid: Callable[[Any], Any] | None = lambda x: x,
        make_element_invalid: Callable[[Any], Any] | None = None) -> None:
    """
    Check whether an object is a moptipy space.

    :param space: the space to test
    :param make_element_valid: a method that can turn a point from the
        space into a valid point
    :param make_element_invalid: a method can a valid point from the
        space into an invalid one
    :raises ValueError: if `space` is not a valid instance of
        :class:`~moptipy.api.space.Space`
    :raises TypeError: if incorrect types are encountered
    """
    if not isinstance(space, Space):
        raise type_error(space, "space", Space)
    check_space(space)
    validate_component(space)

    if not (hasattr(space, "create") and callable(getattr(space, "create"))):
        raise ValueError("space must have method create.")
    x1 = space.create()
    if x1 is None:
        raise ValueError("Spaces must create() valid objects, "
                         "but returned None.")
    x2 = space.create()
    if x2 is None:
        raise ValueError("Spaces must create() valid objects, "
                         "but returned None.")
    if x2 is x1:
        raise ValueError("The create() method must produce different "
                         "instances when invoked twice, but returned the "
                         "same object.")

    if type(x1) is not type(x2):
        raise ValueError("The create() method must produce instances of "
                         f"the same type, but got {type(x1)} and {type(x2)}.")

    if not (hasattr(space, "copy")
            and callable(getattr(space, "copy"))):
        raise ValueError("space must have method copy.")
    space.copy(x2, x1)

    if not (hasattr(space, "is_equal")
            and callable(getattr(space, "is_equal"))):
        raise ValueError("space must have method is_equal.")
    if not space.is_equal(x1, x2):
        raise ValueError("space.copy(x1, x2) did not lead to "
                         "space.is_equal(x1, x2).")

    if make_element_valid is None:
        return

    x1 = make_element_valid(x1)

    if not (hasattr(space, "validate")
            and callable(getattr(space, "validate"))):
        raise ValueError("space must have method validate.")
    space.validate(x1)

    if not (hasattr(space, "to_str") and callable(getattr(space, "to_str"))):
        raise ValueError("space must have method to_str.")
    strstr = space.to_str(x1)
    if not isinstance(strstr, str):
        raise type_error(strstr, f"space.to_str(x) for {x1}", str)
    if len(strstr) <= 0:
        raise ValueError(
            "space.to_str(x) must not produce empty strings, "
            f"but we got '{strstr}'.")
    if strstr.strip() != strstr:
        raise ValueError(
            "space.to_str(x) must not include leading or trailing spaces,"
            f" but we go '{strstr}'.")
    if SECTION_START in strstr:
        raise ValueError(f"space.to_str() must not include "
                         f"'{SECTION_START}', but is '{strstr}'.")
    if SECTION_END in strstr:
        raise ValueError(f"space.to_str() must not include "
                         f"'{SECTION_END}', but is '{strstr}'.")
    if COMMENT_CHAR in strstr:
        raise ValueError(f"space.to_str() must not include "
                         f"'{COMMENT_CHAR}', but is '{strstr}'.")

    if not (hasattr(space, "from_str")
            and callable(getattr(space, "from_str"))):
        raise ValueError("space must have method from_str.")
    x3 = space.from_str(strstr)
    if (x3 is x1) or (x3 is x2):
        raise ValueError("from_str() cannot return the same object as "
                         "create().")
    if not space.is_equal(x1, x3):
        raise ValueError("from_str(to_str()) must return equal object.")
    if space.to_str(x3) != strstr:
        raise ValueError("to_str(from_str(to_str())) must return same "
                         "string.")
    space.validate(x3)

    if make_element_invalid is None:
        return
    x2 = make_element_invalid(x3)
    if space.is_equal(x1, x2):
        raise ValueError(
            "make_element_invalid did not lead to a change in element!")

    with raises(ValueError):
        space.validate(x2)
