"""Functions that can be used to test spaces."""
from typing import Callable, Optional

# noinspection PyProtectedMember
from moptipy.api.space import Space, _check_space
from moptipy.tests.component import check_component


def check_space(space: Space,
                make_valid: Optional[Callable] = lambda x: x) -> None:
    """
    Check whether an object is a moptipy space.
    :param space: the space to test
    :param make_valid: a method that can turn a point from the space into
    a valid point
    :raises ValueError: if `space` is not a valid Space
    """
    if not isinstance(space, Space):
        raise ValueError("Expected to receive an instance of Space, but "
                         "got a '" + str(type(space)) + "'.")
    _check_space(space)
    check_component(component=space)

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

    if not (type(x1) is type(x2)):
        raise ValueError("The create() method must produce instances of "
                         "the same type, but got '" + str(type(x1))
                         + "' and '" + str(type(x2)) + "'.")

    space.copy(x1, x2)
    if not space.is_equal(x1, x2):
        raise ValueError("space.copy(x1, x2) did not lead to "
                         "space.is_equal(x1, x2).")

    scale = space.scale()
    if not isinstance(scale, int):
        raise ValueError("The scale of a space must always be a int, "
                         "but is a '" + str(type(scale)) + "'.")
    if scale <= 0:
        raise ValueError("The scale of a space must be positive, but was "
                         + str(scale) + ".")

    if make_valid is None:
        return

    x1 = make_valid(x1)
    space.validate(x1)

    strstr = space.to_str(x1)
    if not isinstance(strstr, str):
        raise ValueError("space.to_str(x) must produce instances of str, "
                         "but created an instance of '" + str(type(strstr))
                         + "'.")
    if len(strstr.strip()) <= 0:
        raise ValueError("space.to_str(x) must not produce strings just "
                         "composed of white space, but we got '"
                         + strstr + "'.")

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
