"""Simple routines for handling numbers and numerical stuff."""

from math import isfinite, inf, gcd, isqrt, nextafter, log, exp, log2
from typing import Union, Final

from moptipy.utils.types import type_error

#: The positive limit for doubles that can be represented exactly as ints.
DBL_INT_LIMIT_P: Final[float] = 9007199254740992.0  # = 1 << 53
#: The negative  limit for doubles that can be represented exactly as ints.
_DBL_INT_LIMIT_N: Final[float] = -DBL_INT_LIMIT_P


def __try_int(val: float) -> Union[int, float]:
    """
    Convert a float to an int without any fancy checks.

    :param val: the flot
    :returns: the float or int
    """
    if _DBL_INT_LIMIT_N <= val <= DBL_INT_LIMIT_P:
        a = int(val)
        if a == val:
            return a
    return val


def try_int(val: Union[int, float]) -> Union[int, float]:
    """
    Attempt to convert a float to an integer.

    This method will convert a floating point number to an integer if the
    floating point number was representing an exact integer. This is the
    case if it has a) no fractional part and b) is in the range
    `-9007199254740992...9007199254740992`, i.e., the range where `+1` and
    `-1` work without loss of precision.

    :param val: the input value
    :return: an `int` if `val` can be represented as `int` without loss of
        precision, `val` otherwise

    >>> print(type(try_int(10.5)))
    <class 'float'>
    >>> print(type(try_int(10)))
    <class 'int'>
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


def try_int_div(a: int, b: int) -> Union[int, float]:
    """
    Try to divide two integers at best precision.

    Floating point divisions can incur some loss of precision. We try
    to avoid this here as much as possible. First, we check if `a` is
    divisible by `b` without any fractional part. If this is true, then
    we can do a pure integer division without loss of any precision.
    Otherwise, we would do a floating point division. This will lead
    the values be converted to floating point numbers, which can incur
    a loss of precision. We therefore first compute the gcd of `a` and `b`
    and integer-divide both numbers by this value. If the gcd is greater than
    one, then this will make the numbers a bit "smaller", which would reduce
    the loss of precision during the conversion. We then do the final floating
    point division. As a very last step, we check if that result could be
    converted back to an integer, which is very unlikely. This could only
    happen if the division itself incured a loss of precision ... but we give
    it a try anyway.

    :param a: the first integer
    :param b: the second integer
    :return: a/b

    >>> print(type(try_int_div(10, 2)))
    <class 'int'>
    >>> print(type(try_int_div(10, 3)))
    <class 'float'>
    """
    if b == 0:
        return inf

    # First try pure integer division, in case `a` is divisible by `b`.
    int_div_test: Final[int] = a // b
    if (int_div_test * b) == a:
        return int_div_test

    # No, pure division does not work.
    # So let us next divide both `a` and `b` by their gcd.
    gc: Final[int] = gcd(a, b)
    a //= gc
    if b == gc:
        return a
    b //= gc

    # Then we can convert them to floating point numbers with less loss
    # of precision.
    val: Final[float] = a / b
    if not isfinite(val):
        raise ValueError(f"Value must be finite, but is {a}/{b}={val}.")
    # Finally, we attempt to convert the result to an integer again.
    # This can only work if there was some loss of precision during the
    # division.
    return __try_int(val)


def try_float_div(a: Union[int, float], b: Union[int, float]) \
        -> Union[int, float]:
    """
    Try to divide two numbers at best precision.

    First, we will check if we can convert the numbers to integers
    without loss of precision via :func:`try_int`. If yes, then
    we go for the maximum-precision integer division via :func:`try_int_div`.
    If no, then we do the normal floating point division and try to convert
    the result to an integer if that can be done without loss of precision.

    :param a: the first number
    :param b: the second number
    :return: a/b

    >>> print(type(try_float_div(10, 2)))
    <class 'int'>
    >>> print(type(try_float_div(10, 3)))
    <class 'float'>
    >>> print(type(try_float_div(10, 0.5)))
    <class 'int'>
    """
    ia = try_int(a)
    ib = try_int(b)
    if isinstance(ia, int) and isinstance(ib, int):
        return try_int_div(ia, ib)
    val: Final[float] = ia / ib
    if not isfinite(val):
        raise ValueError(f"Value must be finite, but is {a}/{b}={val}.")
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
                 none_on_overflow: bool = True) -> Union[int, float, None]:
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
        raise ValueError(f"huh? {int_root}**{power}={root_power} > {value}?")

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
        raise ValueError(f"huh? {int_root}**{power}={root_power} > {value}?")

    # from now on, root may be either and int or (more likely) a float.
    root: Union[int, float] = int_root
    try:
        rest: Union[int, float] = try_int_div(value, root_power)
        rest_root: Union[int, float] = __try_int(rest ** (1.0 / power))
        root = __try_int(root * rest_root)
    except OverflowError:  # as ofe:
        if none_on_overflow:
            return None
        raise  # raises ofe again

    # OK, we got an approximate root of what remains of value.
    # Let's see if we can refine it.
    try:
        diff = abs((root ** power) - value)
        root2: Union[int, float] = __try_int(exp(log(value) / root))
        diff2: Union[int, float] = abs((root2 ** power) - value)
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

    except OverflowError:
        pass

    return root_base * __try_int(root)
