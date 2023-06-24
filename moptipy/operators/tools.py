"""Some tools for optimization."""

from math import exp, log

import numba  # type: ignore


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def exponential_step_size(step_size: float,
                          min_steps: int, max_steps: int) -> int:
    """
    Translate a step size in `[0,1]` to an integer in `min_steps..max_steps`.

    The idea is that we would like to have step-size dependent operators of
    type :class:`~moptipy.api.operators.Op1WithStepSize`. These operators
    allow algorithms to tune the amount of change to be applied to a solution
    between `0` and `1` (inclusively). As described in the documentation of
    :meth:`~moptipy.api.operators.Op1WithStepSize.op1`, `0` means that the
    smallest possible change should be applied and `1` means that the largest
    possible change should be applied.

    Now for many search spaces, we need to translate this step size from
    `[0,1]` to an integer. For instance, if we have `n`-dimensional
    :mod:`~moptipy.spaces.bitstrings` as search space, then we can flip
    anything between `1` and `n` bits. Straightforwardly, one would linearly
    scale the step size from `[0,1]` to `1..n`. Unfortunately, if we do that,
    then different values of `step_size` will have very different meaning
    depending on `n`. For example, `step_size=0.05` would translate to
    `round(1 + step_size * (n-1)) = 1` bits to be flipped if `n=10`, to
    `6` bit flips for `n=100`, and to `501` bit flips for `n=10_000`. While
    flipping one bit is a very small move and flipping six bits may be a
    medium-size move in discrete optimization, flipping over 500 bits is
    actually always quite a lot.

    What we would like is to have scale-independent small moves but still be
    able to make large moves. We can get this by exponentially transforming
    the `step_width`. Most `step_size` values will result in small integer
    steps and only `step_size` values close to `1` will yield really big
    results.

    And for a `step_size=0.05`, we get for different `n`:

    >>> exponential_step_size(0.05, 1, 10)
    1
    >>> exponential_step_size(0.05, 1, 100)
    1
    >>> exponential_step_size(0.05, 1, 10_000)
    2
    >>> exponential_step_size(0.05, 1, 1_000_000)
    2
    >>> exponential_step_size(0.05, 1, 1_000_000_000)
    3
    >>> exponential_step_size(0.05, 1, 1_000_000_000_000)
    4

    For different values of `step_size` and a fixed `n`, we can still obtain
    the whole spectrum of possible changes. For `n=10`, for example, we get:

    >>> exponential_step_size(0.0, 1, 10)
    1
    >>> exponential_step_size(0.1, 1, 10)
    1
    >>> exponential_step_size(0.2, 1, 10)
    2
    >>> exponential_step_size(0.3, 1, 10)
    2
    >>> exponential_step_size(0.4, 1, 10)
    3
    >>> exponential_step_size(0.5, 1, 10)
    3
    >>> exponential_step_size(0.6, 1, 10)
    4
    >>> exponential_step_size(0.7, 1, 10)
    5
    >>> exponential_step_size(0.8, 1, 10)
    6
    >>> exponential_step_size(0.85, 1, 10)
    7
    >>> exponential_step_size(0.9, 1, 10)
    8
    >>> exponential_step_size(0.95, 1, 10)
    9
    >>> exponential_step_size(1.0, 1, 10)
    10

    So we can still reach the whole range of possible steps from `1` to `n`.

    >>> isinstance(exponential_step_size(0.5, 1, 9), int)
    True
    >>> exponential_step_size(0.0, 1, 100)
    1
    >>> exponential_step_size(1.0, 1, 100)
    100
    >>> exponential_step_size(1.0 / 3.0, 1, 10)
    2
    >>> exponential_step_size(1.0 / 3.0, 1, 100)
    5
    >>> exponential_step_size(1.0 / 3.0, 1, 10_000)
    22
    >>> exponential_step_size(0.0, 2, 10)
    2
    >>> exponential_step_size(0.0, 9, 10)
    9
    >>> exponential_step_size(0.0, 10, 10)
    10
    >>> exponential_step_size(1.0, 10, 10)
    10

    :param step_size: the step size from `[0,1]` to be transformed to an
        integer
    :param min_steps: the minimum (inclusive) value for the returned integer
    :param max_steps: the maximum (inclusive) value for the returned integer
    """
    return round(min_steps + exp(step_size * log(
        max_steps - min_steps + 1))) - 1


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def inv_exponential_step_size(int_val: int,
                              min_steps: int, max_steps: int) -> float:
    """
    Compute the inverse of :func:`exponential_step_size`.

    This routine exists mainly to make testing easier.

    :param int_val: the return value of :func:`exponential_step_size`.
    :param min_steps: the minimum (inclusive) value any `int_val`
    :param max_steps: the maximum (inclusive) value any `int_val`

    >>> exponential_step_size(0.47712125471966244, 1, 10)
    3
    >>> inv_exponential_step_size(3, 1, 10)
    0.47712125471966244
    >>> inv_exponential_step_size(1, 1, 10)
    0.0
    >>> inv_exponential_step_size(10, 1, 10)
    1.0
    >>> inv_exponential_step_size(33, 6, 673)
    0.5123088678224029
    >>> exponential_step_size(0.5123088678224029, 6, 673)
    33
    >>> inv_exponential_step_size(3, 3, 3)
    1.0
    >>> inv_exponential_step_size(3, 3, 10)
    0.0
    >>> inv_exponential_step_size(10, 3, 10)
    1.0
    """
    if int_val >= max_steps:
        return 1.0
    if int_val <= min_steps:
        return 0.0
    return log(int_val - min_steps + 1) / log(max_steps - min_steps + 1)
