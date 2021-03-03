from abc import abstractmethod
from moptipy.api.component import Component
from typing import Optional


class Encoding(Component):
    """
    A class to implement encodings, i.e., functions translating a
    search space to a solution space.
    """

    @abstractmethod
    def map(self, x, y):
        """
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
    An internal method used for checking whether an object is a valid instance
    of :class:`Encoding`
    :param encoding: the object
    :param bool none_is_ok: is it ok if `None` is passed in?
    :return: the object
    :raises ValueError: if `encoding` is not an instance of
    :class:`Encoding`
    """
    if encoding is None:
        if none_is_ok:
            return None
        raise ValueError("This encoding must not be None.")
    if not isinstance(encoding, Encoding):
        raise TypeError(
            "An encoding must be instance of Encoding, but is "
            + str(type(encoding)) + ".")
    return encoding
