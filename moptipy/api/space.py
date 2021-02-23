from abc import abstractmethod
from moptipy.api.component import Component


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
    def x_create(self):
        """
        Generate an instance of the data structure managed by the space.

        The state/contents of this data structure are undefined.

        :return: the new instance
        """
        raise NotImplementedError

    @abstractmethod
    def x_copy(self, source, dest):
        """
        Copy one instance of the data structure to another one.

        :param source: the source data structure, which remains
            unchanged and whose contents will be copied to `dest`
        :param dest: the destination data structure,
            whose contents will be overwritten with those from `source`
        """
        raise NotImplementedError

    @abstractmethod
    def x_to_str(self, x) -> str:
        """
        Obtain a textual representation of an instance of the data structure.

        :param x: the instance
        :return: the string representation of x
        :rtype: str
        """
        raise NotImplementedError

    @abstractmethod
    def x_is_equal(self, x1, x2) -> bool:
        """
        Check if the contents of two instances of the data structure are equal.

        :param x1: the first instance
        :param x2: the second instance
        :return: True if the contents are equal, False otherwise
        :rtype: bool
        """
        raise NotImplementedError
