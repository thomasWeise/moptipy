"""Simple routines for handling numbers and numerical stuff."""

import contextlib
from math import exp, inf, isfinite, isqrt, log, log2, nextafter
from typing import Final

from moptipy.utils.types import type_error

#: The positive limit for doubles that can be represented exactly as ints.
DBL_INT_LIMIT_P: Final[float] = 9007199254740992.0  # = 1 << 53
#: The negative  limit for doubles that can be represented exactly as ints.
_DBL_INT_LIMIT_N: Final[float] = -DBL_INT_LIMIT_P


def __try_int(val: float) -> int | float:
    """
    Convert a float to an int without any fancy checks.

    :param val: the flot
    :returns: the float or int

    >>> from math import inf, nan, nextafter
    >>> type(__try_int(0.0))
    <class 'int'>
    >>> type(__try_int(0.5))
    <class 'float'>
    >>> type(__try_int(inf))
    <class 'float'>
    >>> type(__try_int(-inf))
    <class 'float'>
    >>> type(__try_int(nan))
    <class 'float'>
    >>> 1 << 53
    9007199254740992
    >>> type(__try_int(9007199254740992.0))
    <class 'int'>
    >>> __try_int(9007199254740992.0)
    9007199254740992
    >>> too_big = nextafter(9007199254740992.0, inf)
    >>> print(too_big)
    9007199254740994.0
    >>> type(__try_int(too_big))
    <class 'float'>
    >>> type(__try_int(-9007199254740992.0))
    <class 'int'>
    >>> __try_int(-9007199254740992.0)
    -9007199254740992
    >>> type(__try_int(-too_big))
    <class 'float'>
    """
    if _DBL_INT_LIMIT_N <= val <= DBL_INT_LIMIT_P:
        a = int(val)
        if a == val:
            return a
    return val


def try_int(val: int | float) -> int | float:
    """
    Attempt to convert a float to an integer.

    This method will convert a floating point number to an integer if the
    floating point number was representing an exact integer. This is the
    case if it has a) no fractional part and b) is in the range
    `-9007199254740992...9007199254740992`, i.e., the range where `+1` and
    `-1` work without loss of precision.

    :param val: the input value, which must either be `int` or `float`
    :return: an `int` if `val` can be represented as `int` without loss of
        precision, `val` otherwise
    :raises TypeError: if `val` is neither an instance of `int` nor of `float`
    :raises ValueError: if `val` is a `float`, but not finite

    >>> print(type(try_int(10.5)))
    <class 'float'>
    >>> print(type(try_int(10)))
    <class 'int'>

    >>> from math import inf, nan, nextafter
    >>> type(try_int(0.0))
    <class 'int'>
    >>> type(try_int(0.5))
    <class 'float'>
    >>> try:
    ...     try_int(inf)
    ... except ValueError as ve:
    ...     print(ve)
    Value must be finite, but is inf.
    >>> try:
    ...     try_int(-inf)
    ... except ValueError as ve:
    ...     print(ve)
    Value must be finite, but is -inf.
    >>> try:
    ...     try_int(nan)
    ... except ValueError as ve:
    ...     print(ve)
    Value must be finite, but is nan.
    >>> try:
    ...     try_int("blab")  # noqa  # type: off
    ... except TypeError as te:
    ...     print(te)
    val should be an instance of any in {float, int} but is str, namely \
'blab'.
    >>> type(try_int(9007199254740992.0))
    <class 'int'>
    >>> try_int(9007199254740992.0)
    9007199254740992
    >>> too_big = nextafter(9007199254740992.0, inf)
    >>> print(too_big)
    9007199254740994.0
    >>> type(try_int(too_big))
    <class 'float'>
    >>> type(try_int(-9007199254740992.0))
    <class 'int'>
    >>> try_int(-9007199254740992.0)
    -9007199254740992
    >>> type(try_int(-too_big))
    <class 'float'>
    """
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if not isfinite(val):
            raise ValueError(f"Value must be finite, but is {val}.")
        if _DBL_INT_LIMIT_N <= val <= DBL_INT_LIMIT_P:
            a = int(val)
            if a == val:
                return a
        return val
    raise type_error(val, "val", (int, float))


def try_int_div(a: int, b: int) -> int | float:
    """
    Try to divide two integers at best precision.

    Floating point divisions can incur some loss of precision. We try
    to avoid this here as much as possible. First, we check if `a` is
    divisible by `b` without any fractional part. If this is true, then
    we can do a pure integer division without loss of any precision.

    Otherwise, we would do a floating point division. This will lead the
    values be converted to floating point numbers, which can incur
    a loss of precision. In the past, we tried to divide both numbers by
    their `gcd` to make them smaller in order to avoid a loss of precision.
    But it seems that Python is already doing something like this internally
    when performing the `a / b` division anyway, so we don't do this anymore.
    However, it is still attempted to convert the result back to an integer.

    :param a: the first integer
    :param b: the second integer
    :return: a/b, either as `int` or as `float` but always a finite value
    :raises OverflowError: if the division must be performed using floating
        point arithmetic, but the result would be too large to fit into a
        `float`
    :raises ZeroDivisionError: if `b==0`

    >>> print(try_int_div(10, 2))
    5
    >>> print(try_int_div(10, -2))
    -5
    >>> print(try_int_div(-10, 2))
    -5
    >>> print(try_int_div(-10, -2))
    5
    >>> print(type(try_int_div(10, 2)))
    <class 'int'>
    >>> print(try_int_div(10, 3))
    3.3333333333333335
    >>> print(try_int_div(-10, 3))
    -3.3333333333333335
    >>> print(try_int_div(10, -3))
    -3.3333333333333335
    >>> print(try_int_div(-10, -3))
    3.3333333333333335
    >>> print(type(try_int_div(10, 3)))
    <class 'float'>
    >>> print(try_int_div(9007199254740992, 1))
    9007199254740992
    >>> print(try_int_div(2109792310235001520128, 234234))
    9007199254740992
    >>> print(try_int_div(2109792310235001520128, 234235))
    9007160801054503
    >>> print(try_int_div(2109792310235001520128, 234233))
    9007237708755818.0
    >>> large = 123456789012345678901234567890123456789012345678901234567\
89012345678901234567890123456789012345678901234567890123456789012345678901234\
56789012345678901234567890123456789012345678901234567890123456789012345678901\
23456789012345678901234567890123456789012345678901234567890123456789012345678\
90123456789012345678901234567890123456789012345678901234567890123456789012345\
67890123456789012345678901234567890123456789012345678901234567890123456789012\
3456789012345678901234567890123456789012345678901234567890123456789012345678\
90123456789012345678901234567890123456789012345678901234567890123456789012345\
678901234567890123456789012345678901234567890
    >>> try:
    ...     large / 1
    ... except OverflowError as oe:
    ...     print(oe)
    integer division result too large for a float
    >>> try_int_div(large, 1)
    123456789012345678901234567890123456789012345678901234567\
89012345678901234567890123456789012345678901234567890123456789012345678901234\
56789012345678901234567890123456789012345678901234567890123456789012345678901\
23456789012345678901234567890123456789012345678901234567890123456789012345678\
90123456789012345678901234567890123456789012345678901234567890123456789012345\
67890123456789012345678901234567890123456789012345678901234567890123456789012\
3456789012345678901234567890123456789012345678901234567890123456789012345678\
90123456789012345678901234567890123456789012345678901234567890123456789012345\
678901234567890123456789012345678901234567890
    >>> try_int_div(large * 7, 1 * 7)
    123456789012345678901234567890123456789012345678901234567\
89012345678901234567890123456789012345678901234567890123456789012345678901234\
56789012345678901234567890123456789012345678901234567890123456789012345678901\
23456789012345678901234567890123456789012345678901234567890123456789012345678\
90123456789012345678901234567890123456789012345678901234567890123456789012345\
67890123456789012345678901234567890123456789012345678901234567890123456789012\
3456789012345678901234567890123456789012345678901234567890123456789012345678\
90123456789012345678901234567890123456789012345678901234567890123456789012345\
678901234567890123456789012345678901234567890
    >>> try:
    ...     try_int_div(large, 7)
    ... except OverflowError as oe:
    ...     print(oe)
    integer division result too large for a float
    >>> try:
    ...     try_int_div(0, 0)
    ... except ZeroDivisionError as zde:
    ...     print(zde)
    integer division or modulo by zero
    >>> try:
    ...     try_int_div(1, 0)
    ... except ZeroDivisionError as zde:
    ...     print(zde)
    integer division or modulo by zero
    >>> try:
    ...     try_int_div(-1, 0)
    ... except ZeroDivisionError as zde:
    ...     print(zde)
    integer division or modulo by zero
    """
    # First try pure integer division, in case `a` is divisible by `b`.
    int_div_test: Final[int] = a // b
    if (int_div_test * b) == a:
        return int_div_test
    # If that does not work, use the floating point division.
    return __try_int(a / b)


def try_float_div(a: int | float, b: int | float) -> int | float:
    """
    Try to divide two numbers at best precision.

    First, we will check if we can convert the numbers to integers
    without loss of precision via :func:`try_int`. If yes, then
    we go for the maximum-precision integer division via :func:`try_int_div`.
    If no, then we do the normal floating point division and try to convert
    the result to an integer if that can be done without loss of precision.

    :param a: the first number
    :param b: the second number
    :return: `a/b`, but always finite

    :raises ValueError: if either one of the arguments or the final result
        would not be finite

    >>> try_float_div(1e180, 1e60)
    1.0000000000000001e+120
    >>> try_float_div(1e60, 1e-60)
    1e+120
    >>> try_float_div(1e14, 1e-1)
    1000000000000000
    >>> try_float_div(1e14, -1e-1)
    -1000000000000000
    >>> try_float_div(-1e14, 1e-1)
    -1000000000000000
    >>> try_float_div(-1e14, -1e-1)
    1000000000000000
    >>> try_float_div(1e15, 1e-1)
    1e+16
    >>> try_float_div(1e15, -1e-1)
    -1e+16
    >>> try_float_div(-1e15, 1e-1)
    -1e+16
    >>> try_float_div(-1e15, -1e-1)
    1e+16
    >>> try_float_div(1e15, 1e-15)
    9.999999999999999e+29

    >>> print(type(try_float_div(10, 2)))
    <class 'int'>
    >>> print(type(try_float_div(10, 3)))
    <class 'float'>
    >>> print(type(try_float_div(10, 0.5)))
    <class 'int'>

    >>> from math import inf, nan
    >>> try:
    ...     try_float_div(1.0, 0)
    ... except ZeroDivisionError as zde:
    ...     print(zde)
    integer division or modulo by zero

    >>> try:
    ...     try_float_div(1.0, -0.0)
    ... except ZeroDivisionError as zde:
    ...     print(zde)
    integer division or modulo by zero

    >>> try:
    ...     try_float_div(inf, 0)
    ... except ValueError as ve:
    ...     print(ve)
    Value must be finite, but is inf.

    >>> try:
    ...     try_float_div(-inf, 0)
    ... except ValueError as ve:
    ...     print(ve)
    Value must be finite, but is -inf.

    >>> try:
    ...     try_float_div(nan, 0)
    ... except ValueError as ve:
    ...     print(ve)
    Value must be finite, but is nan.

    >>> try:
    ...     try_float_div(1, inf)
    ... except ValueError as ve:
    ...     print(ve)
    Value must be finite, but is inf.

    >>> try:
    ...     try_float_div(1, -inf)
    ... except ValueError as ve:
    ...     print(ve)
    Value must be finite, but is -inf.

    >>> try:
    ...     try_float_div(1, nan)
    ... except ValueError as ve:
    ...     print(ve)
    Value must be finite, but is nan.

    >>> try:
    ...     try_float_div(1e300, 1e-60)
    ... except ValueError as ve:
    ...     print(ve)
    Result must be finite, but is 1e+300/1e-60=inf.
    """
    ia: Final[int | float] = try_int(a)
    ib: Final[int | float] = try_int(b)
    if isinstance(ia, int) and isinstance(ib, int):
        return try_int_div(ia, ib)
    val: Final[float] = ia / ib
    if not isfinite(val):
        raise ValueError(f"Result must be finite, but is {a}/{b}={val}.")
    return __try_int(val)


def __bin_search_root(value: int, power: int) -> int:
    """
    Search a truncated integer root via binary search.

    :param value: the integer value to compute the root of
    :param power: the integer power of the root
    :returns: the integer root, truncated if necessary
    """
    int_sqrt: Final[int] = isqrt(value)
    if power == 2:
        return int_sqrt

    # compute the maximum base root
    root_min: int = 1
    root_max: int = (int_sqrt - 1) if value > 3 else 1
    while root_max >= root_min:
        root_mid = isqrt(root_min * root_max)
        power_mid = root_mid ** power
        if power_mid > value:
            root_max = root_mid - 1
        elif power_mid < value:
            root_min = root_mid + 1
        else:
            return root_mid
    return root_max


def try_int_root(value: int, power: int,
                 none_on_overflow: bool = True) -> int | float | None:
    """
    Compute `value**(1/power)` where `value` and `power` are both integers.

    :param value: the integer value to compute the root of
    :param power: the integer power of the root
    :param none_on_overflow: `True` if `None` should be returned if we
        encounter an :class:`OverflowError`, `False` if we should simply
        re-raise it
    :returns: the root, or `None` if we encounter an overflow

    >>> try_int_root(100, 2)
    10
    >>> try_int_root(100, 3) - (100 ** (1/3))
    0.0
    >>> try_int_root(123100, 23) - (123100 ** (1/23))
    0.0
    >>> 123100**1000 - try_int_root(123100**1000, 1000)**1000
    0
    >>> 11**290 - try_int_root(11**290, 290)**290
    0
    >>> (abs(1.797E308 - try_int_root(int(1.797E308), 10) ** 10)
    ...     < abs(1.797E308 - (int(1.797E308) ** 0.1) ** 10))
    True
    """
    if not isinstance(value, int):
        raise type_error(value, "value", int)
    if not isinstance(power, int):
        raise type_error(power, "power", int)
    if power <= 0:
        raise ValueError(f"power must be positive but is {power}.")
    if value <= 1:
        if value < 0:
            raise ValueError(f"value must not be negative, but is {value}.")
        return value
    if power == 1:
        return value

    # first, approximate root with crude binary search
    int_root: int = __bin_search_root(value, power)
    root_power: int = int_root ** power
    if root_power >= value:
        if root_power == value:
            return int_root  # ok, we are done
        raise ValueError(f"{int_root}**{power}={root_power} > {value}?")

    # Now try to remove integer factors from the value and the root,
    # to achieve a more accurate division of the rest.
    # This is probably extremely inefficient.
    root_base: int = 1
    i: int = 2
    end: int = min(10_000, int_root)
    while i < end:
        div: int = i ** power
        if (value % div) == 0:
            root_base *= i
            value //= div
            int_root = (int_root + i - 1) // i
            if int_root < end:
                end = int_root
        else:
            i += 1

    # value is now reduced by any integer factor that we can pull out
    # of the root. These factors are aggregated in root_base.
    # If root_base != 1, we need to re-compute the root of the remainder
    # inside value. Whatever root we now compute of value, we must multiply
    # it with root_base.
    if root_base != 1:
        int_root = __bin_search_root(value, power)

    # check again
    root_power = int_root ** power
    if root_power >= value:
        if root_power == value:
            return root_base * int_root  # ok, we are done
        raise ValueError(f"{int_root}**{power}={root_power} > {value}?")

    # from now on, root may be either and int or (more likely) a float.
    root: int | float = int_root
    try:
        rest: int | float = try_int_div(value, root_power)
        rest_root: int | float = __try_int(rest ** (1.0 / power))
        root = __try_int(root * rest_root)
    except OverflowError:  # as ofe:
        if none_on_overflow:
            return None
        raise  # raises ofe again

    # OK, we got an approximate root of what remains of value.
    # Let's see if we can refine it.
    with contextlib.suppress(OverflowError):
        diff = abs((root ** power) - value)
        root2: int | float = __try_int(exp(log(value) / root))
        diff2: int | float = abs((root2 ** power) - value)
        if diff2 < diff:
            diff = diff2
            root = root2
        root2 = __try_int(2 ** (log2(value) / root))
        diff2 = abs((root2 ** power) - value)
        if diff2 < diff:
            diff = diff2
            root = root2

        rdn = root
        rup = root
        while True:
            rdn = nextafter(rdn, -inf)
            apd = abs((rdn ** power) - value)
            if apd > diff:
                break
            diff = apd
            root = rdn
        while True:
            rup = nextafter(rup, inf)
            apd = abs((rup ** power) - value)
            if apd > diff:
                break
            diff = apd
            root = rup

    return root_base * __try_int(root)
