"""The base classes for implementing search operators."""
from numpy.random import Generator

from moptipy.api.component import Component


# start op0
class Op0(Component):
    """A base class to implement a nullary search operator."""

    def op0(self, random: Generator, dest) -> None:
        """
        Apply the nullary search operator to fill object `dest`.

        Afterwards `dest` will hold a valid point in the search space.

        :param Generator random: the random number generator
        :param dest: the destination data structure
        """
# end op0


def check_op0(op0: Op0) -> Op0:
    """
    Check whether an object is a valid instance of :class:`Op0`.

    :param moptipy.api.operators.Op0 op0: the op0 object
    :return: the object
    :raises TypeError: if `op0` is not an instance of :class:`Op0`
    """
    if op0 is None:
        raise TypeError("An Op0 must not be None.")
    if not isinstance(op0, Op0):
        raise TypeError("An Op0 must be instance of Op0, "
                        f"but is {type(op0)}.")
    return op0


# start op1
class Op1(Component):
    """A base class to implement a unary search operator."""

    def op1(self, random: Generator, x, dest) -> None:
        """
        Turn `dest` into a modified copy of `x`.

        :param Generator random: the random number generator
        :param x: the source point in the search space
        :param dest: the destination data structure
        """
# end op1


def check_op1(op1: Op1) -> Op1:
    """
    Check whether an object is a valid instance of :class:`Op1`.

    :param moptipy.api.operators.Op1 op1: the op1 object
    :return: the object
    :raises TypeError: if `op1` is not an instance of :class:`Op1`
    """
    if op1 is None:
        raise TypeError("An Op1 must not be None.")
    if not isinstance(op1, Op1):
        raise TypeError("An Op1 must be instance of Op1, "
                        f"but is {type(op1)}.")
    return op1


# start op2
class Op2(Component):
    """A base class to implement a binary search operator."""

    def op2(self, random: Generator, x0, x1, dest) -> None:
        """
        Fill `dest` with a combination of `x0` and `x1`.

        :param Generator random: the random number generator
        :param x0: the first source point in the search space
        :param x1: the second source point in the search space
        :param dest: the destination data structure
        """
# end op2


def check_op2(op2: Op2) -> Op2:
    """
    Check whether an object is a valid instance of :class:`Op2`.

    :param moptipy.api.operators.Op2 op2: the op2 object
    :return: the object
    :raises TypeError: if `op2` is not an instance of :class:`Op2`
    """
    if op2 is None:
        raise TypeError("An Op2 must not be None.")
    if not isinstance(op2, Op2):
        raise TypeError("An Op2 must be instance of Op2, "
                        f"but is {type(op2)}.")
    return op2
