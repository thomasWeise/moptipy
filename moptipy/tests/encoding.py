"""Functions that can be used to test encodings."""
from typing import Callable, Optional, Any

from moptipy.api.encoding import Encoding, check_encoding
from moptipy.api.space import Space
from moptipy.tests.component import validate_component


def validate_encoding(encoding: Encoding,
                      search_space: Optional[Space] = None,
                      solution_space: Optional[Space] = None,
                      make_search_space_element_valid:
                      Optional[Callable[[Any], Any]] = lambda x: x,
                      is_deterministic: bool = True) -> None:
    """
    Check whether an object is a proper moptipy encoding.

    :param encoding: the encoding to test
    :param search_space: the search space
    :param make_search_space_element_valid: a method that can turn a point
        from the space into a valid point
    :param solution_space: the solution space
    :param is_deterministic: is the mapping deterministic?
    :raises ValueError: if `encoding` is not a valid
        :class:`~moptipy.api.encoding.Encoding`
    """
    if not isinstance(encoding, Encoding):
        raise ValueError("Expected to receive an instance of Encoding, but "
                         f"got a {type(encoding)}.")
    check_encoding(encoding)
    validate_component(encoding)

    count: int = 0
    if search_space is not None:
        count += 1
    if make_search_space_element_valid is not None:
        count += 1
    if solution_space is not None:
        count += 1
    if count <= 0:
        return
    if count < 3:
        raise ValueError(
            "either provide all of search_space, "
            "make_search_space_element_valid, and solution_space or none.")

    x1 = search_space.create()
    if x1 is None:
        raise ValueError("Provided search space created None?")
    x1 = make_search_space_element_valid(x1)
    search_space.validate(x1)

    y1 = solution_space.create()
    if y1 is None:
        raise ValueError("Provided solution space created None?")

    if not (hasattr(encoding, 'map') and callable(getattr(encoding, 'map'))):
        raise ValueError("encoding must have method map.")

    encoding.map(x1, y1)
    solution_space.validate(y1)

    y2 = solution_space.create()
    if y2 is None:
        raise ValueError("Provided solution space created None?")
    if y1 is y2:
        raise ValueError("Provided solution space created "
                         "identical points?")
    encoding.map(x1, y2)
    solution_space.validate(y2)

    if is_deterministic and (not solution_space.is_equal(y1, y2)):
        raise ValueError("Encoding must be deterministic and map "
                         "identical points to same result.")

    x2 = search_space.create()
    if x2 is None:
        raise ValueError("Provided search space created None?")
    if x1 is x2:
        raise ValueError("Provided search space created "
                         "identical points?")

    search_space.copy(x2, x1)
    if not search_space.is_equal(x1, x2):
        raise ValueError("Copy method of search space did not result in "
                         "is_equal becoming true?")
    search_space.validate(x2)

    encoding.map(x2, y2)
    solution_space.validate(y2)

    if is_deterministic and not solution_space.is_equal(y1, y2):
        raise ValueError("Encoding must be deterministic and map "
                         "equal points to same result.")
