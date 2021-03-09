"""This module provides the base class for implementing encodings."""
from abc import abstractmethod
from typing import Optional

from moptipy.api.component import Component


class Encoding(Component):
    """The encodings translates from a search space to a solution space."""

    @abstractmethod
    def map(self, x, y) -> None:
        """
        Translate from search- to solution space.

        Map a point `x` from the search space to a point `y`
        in the solution space.

        :param x: the point in the search space, remaining unchanged.
        :param y: the destination data structure for the point in the
            solution space, whose contents will be overwritten
        """
        raise NotImplementedError


def _check_encoding(encoding: Optional[Encoding],
                    none_is_ok: bool = True) -> Optional[Encoding]:
    """
    Check whether an object is a valid instance of :class:`Encoding`.

    :param encoding: the object
    :param bool none_is_ok: is it ok if `None` is passed in?
    :return: the object
    :raises TypeError: if `encoding` is not an instance of :class:`Encoding`
    """
    if encoding is None:
        if none_is_ok:
            return None
        raise TypeError("The encoding must not be None.")
    if not isinstance(encoding, Encoding):
        raise TypeError(
            "An encoding must be instance of Encoding, but encountered"
            f" {type(encoding)}.")
    return encoding
