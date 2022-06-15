"""The base classes for implementing search operators."""
from numpy.random import Generator

from moptipy.api.component import Component
from moptipy.utils.types import type_error


# start op0
class Op0(Component):
    """A base class to implement a nullary search operator."""

    def op0(self, random: Generator, dest) -> None:
        """
        Apply the nullary search operator to fill object `dest`.

        Afterwards `dest` will hold a valid point in the search space.

        :param random: the random number generator
        :param dest: the destination data structure
        """
        raise ValueError("Method not implemented!")
# end op0


def check_op0(op0: Op0) -> Op0:
    """
    Check whether an object is a valid instance of :class:`Op0`.

    :param op0: the op0 object
    :return: the object
    :raises TypeError: if `op0` is not an instance of :class:`Op0`
    """
    if not isinstance(op0, Op0):
        raise type_error(op0, "op0", Op0)
    return op0


# start op1
class Op1(Component):
    """A base class to implement a unary search operator."""

    def op1(self, random: Generator, dest, x) -> None:
        """
        Turn `dest` into a modified copy of `x`.

        :param random: the random number generator
        :param dest: the destination data structure
        :param x: the source point in the search space
        """
        raise ValueError("Method not implemented!")
# end op1


def check_op1(op1: Op1) -> Op1:
    """
    Check whether an object is a valid instance of :class:`Op1`.

    :param op1: the op1 object
    :return: the object
    :raises TypeError: if `op1` is not an instance of :class:`Op1`
    """
    if not isinstance(op1, Op1):
        raise type_error(op1, "op1", Op1)
    return op1


# start op2
class Op2(Component):
    """A base class to implement a binary search operator."""

    def op2(self, random: Generator, dest, x0, x1) -> None:
        """
        Fill `dest` with a combination of `x0` and `x1`.

        :param random: the random number generator
        :param dest: the destination data structure
        :param x0: the first source point in the search space
        :param x1: the second source point in the search space
        """
        raise ValueError("Method not implemented!")
# end op2


def check_op2(op2: Op2) -> Op2:
    """
    Check whether an object is a valid instance of :class:`Op2`.

    :param op2: the op2 object
    :return: the object
    :raises TypeError: if `op2` is not an instance of :class:`Op2`
    """
    if not isinstance(op2, Op2):
        raise type_error(op2, "op2", Op2)
    return op2
