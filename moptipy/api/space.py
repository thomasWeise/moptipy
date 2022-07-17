"""
Provide the functionality to access search and solution spaces.

A :class:`Space` is the abstraction of the data structures for solutions and
points in the search space that need to be generated, copied, and stored
during the optimization process.
"""
from typing import Optional

from moptipy.api.component import Component
from moptipy.utils.types import type_error


# start book
class Space(Component):
    """
    A class to represent both search and solution spaces.

    The space basically defines a container data structure and basic
    operations that we can apply to them. For example, a solution
    space contains all the possible solutions to an optimization
    problem. All of them are instances of one data structure. An
    optimization as well as a black-box process needs to be able to
    create and copy such objects. In order to store the solutions we
    found in a text file, we must further be able to translate them to
    strings. We should also be able to parse such strings. It is also
    important to detect whether two objects are the same and whether
    the contents of an object are valid. All of this functionality is
    offered by the `Space` class.
    """

    def create(self):
        # end book
        """
        Generate an instance of the data structure managed by the space.

        The state/contents of this data structure are undefined. It may
        not pass the :meth:`validate` method.

        :return: the new instance
        """

    def copy(self, dest, source) -> None:  # +book
        """
        Copy one instance of the data structure to another one.

        :param dest: the destination data structure,
            whose contents will be overwritten with those from `source`
        :param source: the source data structure, which remains
            unchanged and whose contents will be copied to `dest`
        """

    def to_str(self, x) -> str:  # +book
        """
        Obtain a textual representation of an instance of the data structure.

        This method should convert an element of the space to a string
        representation that is parseable by :meth:from_str: and should ideally
        not be too verbose. For example, when converting a list or array `x`
        of integers to a string, one could simply do
        `";".join([str(xx) for xx in x])`, which would convert it to a
        semicolon-separated list without any wasted space.

        :param x: the instance
        :return: the string representation of x
        """

    def is_equal(self, x1, x2) -> bool:  # +book
        """
        Check if the contents of two instances of the data structure are equal.

        :param x1: the first instance
        :param x2: the second instance
        :return: `True` if the contents are equal, `False` otherwise
        """

    def from_str(self, text: str):  # +book
        """
        Transform a string `text` to one element of the space.

        This method should be implemented as inverse to :meth:to_str:.
        It should check the validity of the result before returning it.
        It may not always be possible to implement this method, but you
        should try.

        :param text: the input string
        :return: the element in the space corresponding to `text`
        """

    def validate(self, x) -> None:  # +book
        """
        Check whether a given point in the space is valid.

        :param x: the point
        :raises ValueError: if the point `x` is invalid
        :raises TypeError: if the point `x` (or one of its elements, if
            applicable) has the wrong data type
        """

    def n_points(self) -> int:
        """
        Get the approximate number of different elements in the space.

        This operation can help us when doing tests of the space API
        implementations. If we know how many points exist in the space,
        then we can judge whether a method that randomly generates
        points is sufficiently random, for instance.

        By default, this method simply returns `2`. If you have a better
        approximation of the size of the space, then you should override it.

        :return: the approximate scale of the space
        """
        return 2


def check_space(space: Optional[Space],
                none_is_ok: bool = False) -> Optional[Space]:
    """
    Check whether an object is a valid instance of :class:`Space`.

    :param space: the object
    :param none_is_ok: is it ok if `None` is passed in?
    :return: the object
    :raises TypeError: if `space` is not an instance of
        :class:`~moptipy.api.space.Space`
    """
    if not isinstance(space, Space):
        if none_is_ok and (space is None):
            return None
        raise type_error(space, "space", Space)
    return space
