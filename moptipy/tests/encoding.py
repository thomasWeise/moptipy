"""Functions that can be used to test encodings."""
from typing import Callable, Optional

# noinspection PyProtectedMember
from moptipy.api.encoding import Encoding, _check_encoding
from moptipy.api.space import Space
from moptipy.tests.component import check_component


def check_encoding(encoding: Encoding,
                   search_space: Optional[Space] = None,
                   solution_space: Optional[Space] = None,
                   make_search_space_valid:
                   Optional[Callable] = lambda x: x) -> None:
    """
    Check whether an object is a proper moptipy encoding.
    :param encoding: the encoding to test
    :param search_space: the search space
    :param make_search_space_valid: a method that can turn a point from the
    space into a valid point
    :param solution_space: the solution space
    :raises ValueError: if `encoding` is not a valid :class:`Space`
    """
    if not isinstance(encoding, Encoding):
        raise ValueError("Expected to receive an instance of Encoding, but "
                         "got a '" + str(type(encoding)) + "'.")
    _check_encoding(encoding)
    check_component(component=encoding)

    if (search_space is None) \
            or (make_search_space_valid is None) \
            or (solution_space is None):
        return

    x1 = search_space.create()
    if x1 is None:
        raise ValueError("Provided search space created None?")
    x1 = make_search_space_valid(x1)
    search_space.validate(x1)
    y1 = solution_space.create()
    if y1 is None:
        raise ValueError("Provided solution space created None?")

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

    if not solution_space.is_equal(y1, y2):
        raise ValueError("Encoding must be deterministic and map "
                         "identical points to same result.")

    x2 = search_space.create()
    if x2 is None:
        raise ValueError("Provided search space created None?")
    if x1 is x2:
        raise ValueError("Provided search space created "
                         "identical points?")

    search_space.copy(x1, x2)
    if not search_space.is_equal(x1, x2):
        raise ValueError("Copy method of search space did not result in "
                         "is_equal becoming true?")
    search_space.validate(x2)

    encoding.map(x2, y2)
    solution_space.validate(y2)

    if not solution_space.is_equal(y1, y2):
        raise ValueError("Encoding must be deterministic and map "
                         "equal points to same result.")
