"""Utilities for mock objects."""

from math import ceil, floor, inf, isfinite, nextafter
from typing import Callable, Final, Sequence, cast

import numpy as np
from numpy.random import Generator

from moptipy.utils.types import type_error

#: The default types to be used for testing.
DEFAULT_TEST_DTYPES: Final[tuple[np.dtype, ...]] = tuple(sorted({
    np.dtype(bdt) for bdt in [
        int, float, np.int8, np.int16, np.uint8, np.uint16, np.int32,
        np.uint32, np.int64, np.uint64, np.float16, np.float32,
        np.float64, np.float128]}, key=lambda dt: (dt.kind, dt.itemsize)))


def _lb_int(lb: int | float) -> int:
    """
    Convert a finite lower bound to an integer.

    :param lb: the lower bound
    :retruns: the integer lower bound

    >>> _lb_int(1)
    1
    >>> type(_lb_int(1))
    <class 'int'>
    >>> _lb_int(1.4)
    2
    >>> type(_lb_int(1.4))
    <class 'int'>
    """
    return lb if isinstance(lb, int) else int(ceil(lb))


def _ub_int(ub: int | float) -> int:
    """
    Convert a finite upper bound to an integer.

    :param ub: the upper bound
    :retruns: the integer upper bound

    >>> _ub_int(1)
    1
    >>> type(_ub_int(1))
    <class 'int'>
    >>> _ub_int(1.4)
    1
    >>> type(_ub_int(1.4))
    <class 'int'>
    """
    return ub if isinstance(ub, int) else int(floor(ub))


def _float_beautify(f: float) -> float:
    """
    Get a slightly beautified float, if possible.

    :param f: the float
    :return: the beautified number
    """
    vb: int = int(round(1000.0 * f))
    r1: float = 0.001 * vb
    l1: int = len(str(r1))

    r2: float = 0.001 * (vb + 1)
    l2: int = len(str(r2))
    if l2 < l1:
        l1 = l2
        r1 = r2

    r2 = 0.001 * (vb - 1)
    l2 = len(str(r2))
    if l2 < l1:
        r1 = r2

    return r1


def _before_int(upper_bound: int | float,
                random: Generator) -> int | None:
    """
    Get an `int` value before the given limit.

    :param upper_bound: the upper bound
    :param random: the generator
    :returns: the value, if it could be generated, `None` otherwise
    """
    if upper_bound >= inf:
        upper_bound = 10000
    elif not isfinite(upper_bound):
        return None
    if upper_bound > 0:
        lov = min(int(0.6 * upper_bound), upper_bound - 22)
    else:
        lov = max(int(upper_bound / 0.6), upper_bound - 22)
    lo: Final[int] = _lb_int(max(lov, -9223372036854775806))
    up: Final[int] = _ub_int(upper_bound)
    if lo >= up:
        return None
    res = int(random.integers(lo, up))
    if res >= upper_bound:
        return None
    return res


def _before_float(upper_bound: int | float,
                  random: Generator) -> float | None:
    """
    Get a `float` value before the given limit.

    :param upper_bound: the upper bound
    :param random: the generator
    :returns: the value, if it could be generated, `None` otherwise
    """
    if upper_bound >= inf:
        upper_bound = 10000.0
    elif not isfinite(upper_bound):
        return None
    ulp = 1E16 * (upper_bound - nextafter(upper_bound, -inf)) \
        if (upper_bound < 0) else 1E-8 if upper_bound <= 0 \
        else 1E16 * (nextafter(upper_bound, inf) - upper_bound)
    if upper_bound > 0.0:
        lo = min(upper_bound * 0.6, upper_bound - ulp)
    else:
        lo = min(upper_bound / 0.6, upper_bound - ulp)
    if (not isfinite(lo)) or (lo >= upper_bound):
        return None
    res = float(random.uniform(lo, upper_bound))

    if (not isfinite(res)) or (res >= upper_bound):
        return None
    resb = _float_beautify(res)
    if isfinite(resb) and (resb < upper_bound):
        return resb
    return res


def _after_int(lower_bound: int | float,
               random: Generator) -> int | None:
    """
    Get an `int` value after the given limit.

    :param lower_bound: the upper bound
    :param random: the generator
    :returns: the value, if it could be generated, `None` otherwise
    """
    if lower_bound <= -inf:
        lower_bound = -10000
    elif not isfinite(lower_bound):
        return None
    if lower_bound > 0:
        uv = max(int(lower_bound / 0.6), lower_bound + 22)
    else:
        uv = max(int(lower_bound * 0.6), lower_bound + 22)
    ub: Final[int] = _ub_int(min(uv, 9223372036854775806))
    lb: Final[int] = _lb_int(lower_bound)
    if lb >= ub:
        return None
    res = int(random.integers(lb, ub))
    if res <= lower_bound:
        return None
    return res


def _after_float(lower_bound: int | float,
                 random: Generator) -> float | None:
    """
    Get a `float` value after the given limit.

    :param lower_bound: the upper bound
    :param random: the generator
    :returns: the value, if it could be generated, `None` otherwise
    """
    if lower_bound <= -inf:
        lower_bound = -10000.0
    elif not isfinite(lower_bound):
        return None
    ulp = 1E16 * (lower_bound - nextafter(lower_bound, -inf)) \
        if (lower_bound < 0) else 1E-8 if lower_bound <= 0 \
        else 1E16 * (nextafter(lower_bound, inf) - lower_bound)
    if lower_bound > 0.0:
        hi = max(lower_bound / 0.6, lower_bound + ulp)
    else:
        hi = max(lower_bound * 0.6, lower_bound + ulp)
    if (not isfinite(hi)) or (hi <= lower_bound):
        return None
    res = float(random.uniform(lower_bound, hi))
    if (not isfinite(res)) or (res <= lower_bound):
        return None
    resb = _float_beautify(res)
    if isfinite(resb) and (resb > lower_bound):
        return resb
    return res


def _between_int(lower_bound: int | float,
                 upper_bound: int | float,
                 random: Generator) -> int | None:
    """
    Compute a number between two others.

    :param lower_bound: the minimum
    :param upper_bound: the maximum
    :param random: the generator
    :returns: the value, if it could be generated, `None` otherwise
    """
    if isfinite(lower_bound):
        if isfinite(upper_bound):
            lb: Final[int] = _lb_int(lower_bound) + 1
            ub: Final[int] = _ub_int(upper_bound)
            if lb < ub:
                return int(random.integers(lb, ub))
            return None
        return _after_int(lower_bound, random)
    if isfinite(upper_bound):
        return _before_int(upper_bound, random)
    return int(random.normal(0, 1000.0))


def _between_float(lower_bound: int | float,
                   upper_bound: int | float,
                   random: Generator) -> float | None:
    """
    Compute a number between two others.

    :param lower_bound: the minimum
    :param upper_bound: the maximum
    :param random: the generator
    :returns: the value, if it could be generated, `None` otherwise
    """
    if isfinite(lower_bound):
        if isfinite(upper_bound):
            a = lower_bound
            b = upper_bound
            for _ in range(5):
                a = nextafter(a, inf)
                b = nextafter(b, -inf)
            if a < b:
                res = max(a, min(b, float(random.uniform(a, b))))
                if not isfinite(res) or not (lower_bound < res < upper_bound):
                    return None
                resb = _float_beautify(res)
                if isfinite(resb) and (lower_bound < resb < upper_bound):
                    return resb
                return res
            return None
        return _after_float(lower_bound, random)
    if isfinite(upper_bound):
        return _before_float(upper_bound, random)
    return float(random.normal(0, 1000.0))


def make_ordered_list(definition: Sequence[int | float | None],
                      is_int: bool, random: Generator) \
        -> list[int | float] | None:
    """
    Make an ordered list of elements, filling in gaps.

    This function takes a list template where some values may be defined
    and some may be left `None`.
    The `None` values are then replaced such that an overall ordered list
    is created where each value is larger than its predecessor.
    The original non-`None` elements are kept in place.
    Of course, this process may fail, in which case `None` is returned.

    :param definition: a template with `None` for gaps to be filled
    :param is_int: should all generated values be integers?
    :param random: the generator
    :returns: the refined tuple with all values filled in

    >>> from numpy.random import default_rng
    >>> rg = default_rng(11)
    >>> make_ordered_list([None, 10, None, None, 50], True, rg)
    [-10, 10, 42, 47, 50]
    >>> make_ordered_list([None, 10, None, None, 50, None, None], False, rg)
    [-5.136, 10, 13.228, 15.19, 50, 115.953, 125.961]
    >>> print(make_ordered_list([9, None, 10, None, None, 50], True, rg))
    None
    >>> make_ordered_list([8, None, 10, None, None, 50], True, rg)
    [8, 9, 10, 45, 47, 50]
    >>> make_ordered_list([9, None, 10, None, None, 50], False, rg)
    [9, 9.568, 10, 47.576, 49.482, 50]
    """
    if not isinstance(definition, Sequence):
        raise type_error(definition, "definition", Sequence)
    total: Final[int] = len(definition)
    if total <= 0:
        return []

    if not isinstance(random, Generator):
        raise type_error(random, "random", Generator)
    if not isinstance(is_int, bool):
        raise type_error(is_int, "is_int", bool)

    if is_int:
        _before = cast(Callable[[int | float, Generator],
                                int | float | None], _before_int)
        _after = cast(Callable[[int | float, Generator],
                               int | float | None], _after_int)
        _between = cast(Callable[[int | float, int | float,
                                  Generator], int | float | None],
                        _between_int)
    else:
        _before = cast(Callable[[int | float, Generator],
                                int | float | None], _before_float)
        _after = cast(Callable[[int | float, Generator],
                               int | float | None], _after_float)
        _between = cast(Callable[[int | float, int | float,
                                  Generator], int | float | None],
                        _between_float)

    max_trials: int = 1000
    while max_trials > 0:
        max_trials = max_trials - 1
        result = list(definition)

        failed: bool = False

        # create one random midpoint if necessary
        has_defined: bool = False
        for i in range(total):
            if result[i] is not None:
                has_defined = True
                break
        if not has_defined:
            val = _between(-inf, inf, random)
            result[random.integers(total)] = val
            failed = val is None
        if failed:
            continue

        # fill front backwards
        for i in range(total):
            ub = result[i]
            if ub is not None:
                for j in range(i - 1, -1, -1):
                    ub = _before(ub, random)
                    if ub is None:
                        failed = True
                        break
                    result[j] = ub
                break
        if failed:
            continue

        # fill end forward
        for i in range(total - 1, -1, -1):
            lb = result[i]
            if lb is not None:
                for j in range(i + 1, total):
                    lb = _after(lb, random)
                    if lb is None:
                        failed = True
                        break
                    result[j] = lb
                break
        if failed:
            continue

        # fill all the gaps in between
        while not failed:
            # find random gap
            has_missing: bool = False
            ofs: int = random.integers(total)
            missing: int = 0
            for i in range(total):
                missing = (ofs + i) % total
                if result[missing] is None:
                    has_missing = True
                    break
            if not has_missing:
                break

            # find start of gap and lower bound
            prev_idx: int = missing
            prev: int | float | None = None
            for i in range(missing - 1, -1, -1):
                prev = result[i]
                if prev is not None:
                    prev_idx = i
                    break

            # find end of gap and upper bound
            nxt_idx: int = missing
            nxt: int | float | None = None
            for i in range(missing + 1, total):
                nxt = result[i]
                if nxt is not None:
                    nxt_idx = i
                    break

            # generate new value and store at random position in gap
            val = _between(prev, nxt, random)
            if val is None:
                failed = True
                break
            result[random.integers(prev_idx + 1, nxt_idx)] = val

        if failed:
            continue

        # now check and return result
        prev = result[0]
        if prev is None:
            continue
        for i in range(1, total):
            nxt = result[i]
            if (nxt is None) or (nxt <= prev):
                failed = True
                break
            prev = nxt
        if not failed:
            return result

    return None


def sample_from_attractors(random: Generator,
                           attractors: Sequence[int | float],
                           is_int: bool = False,
                           lb: int | float = -inf,
                           ub: int | float = inf) -> int | float:
    """
    Sample from a given range using the specified attractors.

    :param random: the random number generator
    :param attractors: the attractor points
    :param lb: the lower bound
    :param ub: the upper bound
    :param is_int: shall we sample integer values?
    :return: the value
    :raises ValueError: if the sampling failed

    >>> from numpy.random import default_rng
    >>> rg = default_rng(11)
    >>> sample_from_attractors(rg, [5, 20])
    15.198106552324713
    >>> sample_from_attractors(rg, [2], lb=0, ub=10, is_int=True)
    3
    >>> sample_from_attractors(rg, [5, 20], lb=4)
    4.7448464616061665
    >>> sample_from_attractors(rg, [5, 20], ub=22)
    1.044618552249311
    >>> sample_from_attractors(rg, [5, 20], lb=0, ub=30, is_int=True)
    6
    >>> sample_from_attractors(rg, [5, 20], lb=4, ub=22, is_int=True)
    20
    """
    max_trials: int = 1000
    al: Final[int] = len(attractors)
    while max_trials > 0:
        max_trials -= 1

        chosen_idx = random.integers(al)
        chosen = attractors[chosen_idx]
        lo = attractors[chosen_idx - 1] if (chosen_idx > 0) else lb
        hi = attractors[chosen_idx + 1] if (chosen_idx < (al - 1)) else ub

        sd = 0.5 * min(hi - chosen, chosen - lo)
        if not isfinite(sd):
            sd = max(1.0, 0.05 * abs(chosen))
        sample = random.normal(chosen, sd)
        if not isfinite(sample):
            continue
        if is_int:
            sample = int(sample)
        else:
            sample = float(sample)
        if lb <= sample <= ub:
            return sample

    raise ValueError(f"Failed to sample with lb={lb}, ub={ub}, "
                     f"attractors={attractors}, is_int={is_int}.")
