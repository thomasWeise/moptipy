"""
The base classes for implementing search operators.

Nullary search operators are used to sample the initial starting points of the
optimization processes. They inherit from class
:class:`~moptipy.api.operators.Op0`. The pre-defined unit test routine
:func:`~moptipy.tests.op0.validate_op0` can and should be used to test all the
nullary operators that are implemented.

Unary search operators accept one point in the search space as input and
generate a new, similar point as output. They inherit from class
:class:`~moptipy.api.operators.Op1`. The pre-defined unit test routine
:func:`~moptipy.tests.op1.validate_op1` can and should be used to test all the
unary operators that are implemented.

The basic unary operators :class:`~moptipy.api.operators.Op1` have no
parameter telling them how much of the input point to change. They may do a
hard-coded number of modifications (as, e.g.,
:class:`~moptipy.operators.permutations.op1_swap2.Op1Swap2` does) or may
apply a random number of modifications (like
:class:`~moptipy.operators.permutations.op1_swapn.Op1SwapN`). There is a
sub-class of unary operators named
:class:`~moptipy.api.operators.Op1WithStepSize` where a parameter `step_size`
with a value from the closed interval `[0.0, 1.0]` can be supplied. If
`step_size=0.0`, such an operator should perform the smallest possible
modification and for `step_size=1.0`, it should perform the largest possible
modification.

Binary search operators accept two points in the search space as input and
generate a new point that should be similar to both inputs as output. They
inherit from class :class:`~moptipy.api.operators.Op2`. The pre-defined unit
test routine :func:`~moptipy.tests.op2.validate_op2` can and should be used to
test all the binary operators that are implemented.
"""
from typing import Any

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
        Often, this would be a point uniformly randomly sampled from the
        search space, but it could also be the result of a heuristic or
        even a specific solution.

        :param random: the random number generator
        :param dest: the destination data structure
        """
        raise ValueError("Method not implemented!")
# end op0


def check_op0(op0: Any) -> Op0:
    """
    Check whether an object is a valid instance of :class:`Op0`.

    :param op0: the (supposed) instance of :class:`Op0`
    :return: the object `op0`
    :raises TypeError: if `op0` is not an instance of :class:`Op0`

    >>> check_op0(Op0())
    Op0
    >>> try:
    ...     check_op0('A')
    ... except TypeError as te:
    ...     print(te)
    op0 should be an instance of moptipy.api.operators.Op0 but is \
str, namely 'A'.
    >>> try:
    ...     check_op0(None)
    ... except TypeError as te:
    ...     print(te)
    op0 should be an instance of moptipy.api.operators.Op0 but is None.
    """
    if isinstance(op0, Op0):
        return op0
    raise type_error(op0, "op0", Op0)


# start op1
class Op1(Component):
    """A base class to implement a unary search operator."""

    def op1(self, random: Generator, dest, x) -> None:
        """
        Fill `dest` with a modified copy of `x`.

        :param random: the random number generator
        :param dest: the destination data structure
        :param x: the source point in the search space
        """
        raise ValueError("Method not implemented!")
# end op1


def check_op1(op1: Any) -> Op1:
    """
    Check whether an object is a valid instance of :class:`Op1`.

    :param op1: the (supposed) instance of :class:`Op1`
    :return: the object
    :raises TypeError: if `op1` is not an instance of :class:`Op1`

    >>> check_op1(Op1())
    Op1
    >>> try:
    ...     check_op1('A')
    ... except TypeError as te:
    ...     print(te)
    op1 should be an instance of moptipy.api.operators.Op1 but is str, \
namely 'A'.
    >>> try:
    ...     check_op1(None)
    ... except TypeError as te:
    ...     print(te)
    op1 should be an instance of moptipy.api.operators.Op1 but is None.
    """
    if isinstance(op1, Op1):
        return op1
    raise type_error(op1, "op1", Op1)


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


def check_op2(op2: Any) -> Op2:
    """
    Check whether an object is a valid instance of :class:`Op2`.

    :param op2: the (supposed) instance of :class:`Op2`
    :return: the object `op2`
    :raises TypeError: if `op2` is not an instance of :class:`Op2`

    >>> check_op2(Op2())
    Op2
    >>> try:
    ...     check_op2('A')
    ... except TypeError as te:
    ...     print(te)
    op2 should be an instance of moptipy.api.operators.Op2 but is str, \
namely 'A'.
    >>> try:
    ...     check_op2(None)
    ... except TypeError as te:
    ...     print(te)
    op2 should be an instance of moptipy.api.operators.Op2 but is None.
    """
    if isinstance(op2, Op2):
        return op2
    raise type_error(op2, "op2", Op2)


# start op1WithStepSize
class Op1WithStepSize(Op1):
    """A unary search operator with a step size."""

    def op1(self, random: Generator, dest, x, step_size: float = 0.0) -> None:
        """
        Copy `x` to `dest` but apply a modification with a given `step_size`.

        This operator is similar to :meth:`Op1.op1` in that it stores a
        modified copy of `x` into `dest`. The difference is that you can also
        specify how much that copy should be different: The parameter
        `step_size` can take on any value in the interval `[0.0, 1.0]`,
        including the two boundary values. A `step_size` of `0.0` indicates
        the smallest possible move (for which `dest` will still be different
        from `x`) and `step_size=1.0` will lead to the largest possible move.

        The `step_size` may be interpreted differently by different operators:
        Some may interpret it as an exact requirement and enforce steps of the
        exact specified size, see, for example module
        :mod:`~moptipy.operators.bitstrings.op1_flip_m`. Others might
        interpret it stochastically as an expectation. Yet others may
        interpret it as a goal step width and try to realize it in a best
        effort kind of way, but may also do smaller or larger steps if the
        best effort fails, see for example module
        :mod:`~moptipy.operators.permutations.op1_swap_exactly_n`.
        What all operators should, however, have in common is that at
        `step_size=0.0`, they should try to perform a smallest possible change
        and at `step_size=1.0`, they should try to perform a largest possible
        change. For all values in between, step sizes should grow with rising
        `step_size`. This should allow algorithms that know nothing about the
        nature of the search space or the operator's moves to still tune
        between small and large moves based on a policy which makes sense in a
        black-box setting.

        Every implementation of :class:`Op1WithStepSize` must specify a
        reasonable default value for this parameter ensure compatibility with
        :meth:`Op1.op1`. In this base class, we set the default to `0.0`.

        Finally, if a `step_size` value is passed in which is outside the
        interval `[0, 1]`, the behavior of this method is undefined. It may
        throw an exception or not. It may also enter an infinite loop.

        :param random: the random number generator
        :param dest: the destination data structure
        :param x: the source point in the search space
        :param step_size: the step size parameter for the unary operator
        """
        raise ValueError("Method not implemented!")
# end op1WithStepSize


def check_op1_with_step_size(op1: Any) -> Op1WithStepSize:
    """
    Check whether an object is a valid instance of :class:`Op1WithStepSize`.

    :param op1: the (supposed) instance of :class:`Op1WithStepSize`
    :return: the object `op1`
    :raises TypeError: if `op1` is not an instance of :class:`Op1WithStepSize`

    >>> check_op1_with_step_size(Op1WithStepSize())
    Op1WithStepSize
    >>> try:
    ...     check_op1_with_step_size('A')
    ... except TypeError as te:
    ...     print(te)
    op1 should be an instance of moptipy.api.operators.Op1WithStepSize \
but is str, namely 'A'.
    >>> try:
    ...     check_op1_with_step_size(Op1())
    ... except TypeError as te:
    ...     print(te)
    op1 should be an instance of moptipy.api.operators.Op1WithStepSize \
but is moptipy.api.operators.Op1, namely 'Op1'.
    >>> try:
    ...     check_op1_with_step_size(None)
    ... except TypeError as te:
    ...     print(te)
    op1 should be an instance of moptipy.api.operators.Op1WithStepSize \
but is None.
    """
    if isinstance(op1, Op1WithStepSize):
        return op1
    raise type_error(op1, "op1", Op1WithStepSize)
