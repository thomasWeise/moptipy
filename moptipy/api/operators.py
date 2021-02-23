from abc import abstractmethod

from numpy.random import Generator

from moptipy.api.component import Component


class Op0(Component):
    """
    A class to implement a nullary search operator.
    """

    @abstractmethod
    def op0(self, random: Generator, dest):
        """
        Apply the nullary search operator to fill object `dest`.
        Afterwards `dest` will hold a valid point in the search space.

        :param Generator random: the random number generator
        :param dest: the destination data structure
        """
        raise NotImplementedError


class Op1(Component):
    """
    A class to implement a unary search operator.
    """

    @abstractmethod
    def op1(self, random: Generator, x, dest):
        """
        Apply the unary search operator to fill object `dest`
        with a modified version of the source `x`.
        Afterwards `dest` will hold a valid point in the search space.

        :param Generator random: the random number generator
        :param x: the source point in the search space
        :param dest: the destination data structure
        """
        raise NotImplementedError


class Op2(Component):
    """
    A class to implement a binary search operator.
    """

    @abstractmethod
    def op2(self, random: Generator, x0, x1, dest):
        """
        Apply the binary search operator to fill object `dest` with
        a combination of the contents of `x0` and `x1`.
        Afterwards `dest` will hold a valid point in the search space.

        :param Generator random: the random number generator
        :param x0: the first source point in the search space
        :param x1: the second source point in the search space
        :param dest: the destination data structure
        """
        raise NotImplementedError
