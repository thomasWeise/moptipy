from abc import abstractmethod
from moptipy.api.component import Component
from typing import Optional


class Space(Component):
    """
    A class to represent both search and solution spaces.

    The space basically defines a container data structure and basic operations
    that we can apply to them. For instance, a solution space contains all the
    possible solutions to an optimization problem. All of them are instances of
    one data structure. An optimization as well as a black-box process needs to
    be able to create and copy such instances. In order to store the solutions
    we found in a log file, we must further be able to translate them to
    strings. Finally, it is also important to detect whether two instances are
    the same. All of this functionality is offered by the Space class.
    """

    @abstractmethod
    def create(self):
        """
        Generate an instance of the data structure managed by the space.

        The state/contents of this data structure are undefined.

        :return: the new instance
        """
        raise NotImplementedError

    @abstractmethod
    def copy(self, source, dest):
        """
        Copy one instance of the data structure to another one.

        :param source: the source data structure, which remains
            unchanged and whose contents will be copied to `dest`
        :param dest: the destination data structure,
            whose contents will be overwritten with those from `source`
        """
        raise NotImplementedError

    @abstractmethod
    def to_str(self, x) -> str:
        """
        Obtain a textual representation of an instance of the data structure.

        :param x: the instance
        :return: the string representation of x
        :rtype: str
        """
        raise NotImplementedError

    @abstractmethod
    def is_equal(self, x1, x2) -> bool:
        """
        Check if the contents of two instances of the data structure are equal.

        :param x1: the first instance
        :param x2: the second instance
        :return: True if the contents are equal, False otherwise
        :rtype: bool
        """
        raise NotImplementedError

    @abstractmethod
    def from_str(self, text: str):
        """
        Transform a string `text` to one element of the space.
        This method should be implemented as inverse to
        :py:meth:x_to_str:.
        It may not always be possible to implement this method.
        :param text: the input string
        :return: the element in the space corresponding to `text`
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, x):
        """
        Check whether a given point in the space is valid
        :param x: the point
        :raises ValueError: if the point `x` is invalid
        """
        raise NotImplementedError

    @abstractmethod
    def scale(self) -> int:
        """
        The approximate size of the space, i.e., the approximate number of
        different points it contains
        :return: the approximate scale of the space
        """
        raise NotImplementedError


def _check_space(space: Optional[Space],
                 none_is_ok: bool = False) -> Optional[Space]:
    """
    An internal method used for checking whether an object is a valid instance
    of :class:`Space`
    :param space: the object
    :param bool none_is_ok: is it ok if `None` is passed in?
    :return: the object
    :raises ValueError: if `space` is not an instance of
    :class:`Space`
    """
    if space is None:
        if none_is_ok:
            return None
        raise ValueError("This space must not be None.")
    if not isinstance(space, Space):
        raise TypeError("A space must be instance of Space, but is "
                        + str(type(space)) + ".")
    return space
