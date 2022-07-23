"""Utilities for interaction with numpy."""
from hashlib import sha512
from math import isfinite
from typing import Final, List, Set, Dict, Tuple, Union, cast

import numba  # type: ignore
import numpy
from numpy.random import default_rng, Generator, PCG64

from moptipy.utils.logger import CSV_SEPARATOR
from moptipy.utils.strings import num_to_str, bool_to_str
from moptipy.utils.types import type_error

#: All the numpy integer types and their ranges in increasing order of size.
#: The tuple contains alternating signed and unsigned types. It starts with
#: the smallest signed type, `numpy.int8` and ends with the largest unsigned
#: type `numpy.uint64`.
#: If we have a range `[min..max]` of valid value, then we can look up this
#: range and find the integer type with the smallest memory footprint to
#: accommodate it. This is what :func:`int_range_to_dtype` does.
__INTS_AND_RANGES: Final[Tuple[Tuple[numpy.dtype, int, int], ...]] = \
    tuple(sorted([
        (dtx, int(numpy.iinfo(dtx).min), int(numpy.iinfo(dtx).max))
        for dtx in cast(Set[numpy.dtype], set(numpy.dtype(bdt) for bdt in [
            int, numpy.int8, numpy.int16, numpy.uint8, numpy.uint16,
            numpy.int32, numpy.uint32, numpy.int64, numpy.uint64]))],
        key=lambda a: (a[2], a[1])))

#: The numpy integer data types.
INTS: Final[Tuple[numpy.dtype, ...]] = tuple(a[0] for a in __INTS_AND_RANGES)

#: A map associating all numpy integer types associated to tuples
#: of their respective minimum and maximum value.
__NP_INTS_MAP: Final[Dict[numpy.dtype, Tuple[int, int]]] = \
    {a[0]: (a[1], a[2]) for a in __INTS_AND_RANGES}

#: The default integer type: the signed 64-bit integer.
DEFAULT_INT: Final[numpy.dtype] = INTS[-2]

#: The default unsigned integer type: an unsigned 64-bit integer.
DEFAULT_UNSIGNED_INT: Final[numpy.dtype] = INTS[-1]

#: The default boolean type.
DEFAULT_BOOL: Final[numpy.dtype] = numpy.dtype(numpy.bool_)

#: The default floating point type.
DEFAULT_FLOAT: Final[numpy.dtype] = numpy.dtype(float)

#: The default numerical types.
DEFAULT_NUMERICAL: Final[Tuple[numpy.dtype, ...]] = tuple(
    list(INTS) + [DEFAULT_FLOAT])


def is_np_int(dtype: numpy.dtype) -> bool:
    """
    Check whether a :class:`numpy.dtype` is an integer type.

    :param dtype: the type

    >>> import numpy as npx
    >>> from moptipy.utils.nputils import is_np_int
    >>> print(is_np_int(npx.dtype(npx.int8)))
    True
    >>> print(is_np_int(npx.dtype(npx.uint16)))
    True
    >>> print(is_np_int(npx.dtype(npx.float64)))
    False
    """
    return dtype.kind in ('i', 'u')


def is_np_float(dtype: numpy.dtype) -> bool:
    """
    Check whether a :class:`numpy.dtype` is a floating point type.

    :param dtype: the type

    >>> import numpy as npx
    >>> from moptipy.utils.nputils import is_np_float
    >>> print(is_np_float(npx.dtype(npx.int8)))
    False
    >>> print(is_np_float(npx.dtype(npx.uint16)))
    False
    >>> print(is_np_float(npx.dtype(npx.float64)))
    True
    """
    return dtype.kind == 'f'


def int_range_to_dtype(min_value: int, max_value: int) -> numpy.dtype:
    """
    Convert an integer range to an appropriate numpy data type.

    The returned type is as compact as possible and signed types are
    preferred over unsigned types.
    The returned :class:`numpy.dtype` will allow accommodating all values in
    the inclusive interval `min_value..max_value`.

    :param min_value: the minimum value
    :param max_value: the maximum value
    :return: the numpy integer range
    :raises TypeError: if the parameters are not integers
    :raises ValueError: if the range is invalid

    >>> from moptipy.utils.nputils import int_range_to_dtype
    >>> print(int_range_to_dtype(0, 127))
    int8
    >>> print(int_range_to_dtype(0, 128))
    uint8
    >>> print(int_range_to_dtype(0, 32767))
    int16
    >>> print(int_range_to_dtype(0, 32768))
    uint16
    >>> print(int_range_to_dtype(0, (2 ** 31) - 1))
    int32
    >>> print(int_range_to_dtype(0, 2 ** 31))
    uint32
    >>> print(int_range_to_dtype(0, (2 ** 63) - 1))
    int64
    >>> print(int_range_to_dtype(0, 2 ** 63))
    uint64
    >>> print(int_range_to_dtype(0, (2 ** 64) - 1))
    uint64
    """
    if not isinstance(min_value, int):
        raise type_error(min_value, "min_value", int)
    if not isinstance(max_value, int):
        raise type_error(max_value, "max_value", int)
    if min_value > max_value:
        raise ValueError(
            f"min_value must be <= max_value, but min_value={min_value} "
            f"and max_value={max_value} was provided.")

    for t in __INTS_AND_RANGES:
        if (min_value >= t[1]) and (max_value <= t[2]):
            return t[0]

    if min_value >= 0:
        raise ValueError(
            "max_value for unsigned integers must be <="
            f"{(__INTS_AND_RANGES[-1])[2]}, but is {max_value} "
            f" for min_value={min_value}.")

    raise ValueError(
        f"Signed integer range cannot exceed {__INTS_AND_RANGES[-2][1]}.."
        f"{__INTS_AND_RANGES[-2][2]}, but {min_value}..{max_value} "
        "was specified.")


def dtype_for_data(always_int: bool,
                   lower_bound: Union[int, float],
                   upper_bound: Union[int, float]) -> numpy.dtype:
    """
    Obtain the most suitable numpy data type to represent the data.

    :param always_int: is the data always integer?
    :param lower_bound: the lower bound of the data, set to `-inf` if no lower
        bound is known and we should assume the full integer range
    :param upper_bound: the upper bound of the data, set to `-inf` if no upper
        bound is known and we should assume the full integer range
    """
    if not isinstance(always_int, bool):
        raise type_error(always_int, "always_int", bool)
    if not isinstance(lower_bound, (int, float)):
        raise type_error(lower_bound, "lower_bound", (int, float))
    if not isinstance(upper_bound, (int, float)):
        raise type_error(upper_bound, "upper_bound", (int, float))
    if always_int:
        if isfinite(lower_bound):
            if not isinstance(lower_bound, int):
                raise type_error(
                    lower_bound, "finite lower_bound of always_int", int)
            if isfinite(upper_bound):
                if not isinstance(upper_bound, int):
                    raise type_error(
                        upper_bound, "finite upper_bound of always_int", int)
                return int_range_to_dtype(int(lower_bound), int(upper_bound))
            if lower_bound >= 0:
                return DEFAULT_UNSIGNED_INT
        return DEFAULT_INT
    return DEFAULT_FLOAT


@numba.jit(forceobj=True)
def np_ints_max(shape, dtype: numpy.dtype = DEFAULT_INT) -> numpy.ndarray:
    """
    Create an integer array of the given length filled with the maximum value.

    :param shape: the requested shape
    :param dtype: the data type (defaults to 64bit integers)
    :return: the new array

    >>> import numpy as npx
    >>> from moptipy.utils.nputils import np_ints_max
    >>> print(np_ints_max(4, npx.dtype("uint8")))
    [255 255 255 255]
    """
    return numpy.full(shape=shape,
                      fill_value=__NP_INTS_MAP[dtype][1],
                      dtype=dtype)


#: the default number of bytes for random seeds
__SEED_BYTES: Final = 8
#: the minimum acceptable random seed
__MIN_RAND_SEED: Final = 0
#: the maximum acceptable random seed
__MAX_RAND_SEED: Final = int((1 << (__SEED_BYTES * 8)) - 1)


def rand_seed_generate(random: Generator = default_rng()) -> int:
    """
    Generate a random seed.

    :param random: the random number generator to be used to generate the seed
    :return: the random seed
    :raises TypeError: if `random` is specified but is not an instance of
        `Generator`
    """
    if not isinstance(random, Generator):
        raise type_error(random, "random", Generator)
    return int.from_bytes(random.bytes(__SEED_BYTES),
                          byteorder='big', signed=False)


def rand_seed_check(rand_seed: int) -> int:
    """
    Make sure that a random seed is valid.

    :param rand_seed: the random seed to check
    :return: the rand seed

    :raises TypeError: if the random seed is not an `int`
    :raises ValueError: if the random seed is not valid
    """
    if not isinstance(rand_seed, int):
        raise type_error(rand_seed, "rand_seed", int)
    if (rand_seed < __MIN_RAND_SEED) or (rand_seed > __MAX_RAND_SEED):
        raise ValueError(f"rand_seed must be in {__MIN_RAND_SEED}.."
                         f"{__MAX_RAND_SEED}, but is {rand_seed}.")
    return rand_seed


def rand_generator(seed: int) -> Generator:
    """
    Instantiate a random number generator from a seed.

    :param seed: the random seed
    :return: the random number generator
    """
    return default_rng(rand_seed_check(seed))


def rand_seeds_from_str(string: str,
                        n_seeds: int) -> List[int]:
    """
    Reproducibly generate `n_seeds` unique random seeds from a `string`.

    This function will produce a sorted sequence of `n_seeds`random seeds,
    each of which being an unsigned 64 bit integer, from the string passed in.
    The same string will always yield the same sequence reproducibly.
    Running the function twice with different values of `n_seeds` will result
    in the two sets of random seeds, where the larger one (for the larger
    value of `n_seeds`) contains all elements of the smaller one.

    :param string: the string
    :param n_seeds: the number of seeds
    :return: a tuple of random seeds
    :raises TypeError: if the parameters do not follow the type contract
    :raises ValueError: if the parameter values are invalid

    >>> rand_seeds_from_str("hello world!", 1)
    [11688012229199056962]
    >>> rand_seeds_from_str("hello world!", 2)
    [3727742416375614079, 11688012229199056962]
    >>> rand_seeds_from_str("hello world!", 3)
    [3727742416375614079, 11688012229199056962, 17315292100125916507]

    >>> rand_seeds_from_str("metaheuristic optimization", 1)
    [12323230366215963648]
    >>> rand_seeds_from_str("metaheuristic optimization", 2)
    [12323230366215963648, 13673960948036381176]
    >>> rand_seeds_from_str("metaheuristic optimization", 3)
    [12323230366215963648, 13673960948036381176, 18426184104943646060]
    """
    if not isinstance(string, str):
        raise type_error(string, "string", str)
    if len(string) <= 0:
        raise ValueError("string must not be empty.")
    if not isinstance(n_seeds, int):
        raise type_error(n_seeds, "n_seeds", int)
    if n_seeds <= 0:
        raise ValueError(
            f"n_seeds must be positive, but is {n_seeds}.")

    seeds = bytearray(sha512(string.encode("utf8")).digest())
    seed1 = int.from_bytes(seeds[0:32], byteorder='big', signed=False)
    seed2 = int.from_bytes(seeds[32:64], byteorder='big', signed=False)
    del seeds

    # seed two PCG64 generators, each of which should use two 128 bit
    # numbers as seed
    g1 = Generator(PCG64(seed1))
    g2 = Generator(PCG64(seed2))

    generated: Set[int] = set()
    while len(generated) < n_seeds:
        g1, g2 = g2, g1
        generated.add(rand_seed_generate(g1))

    result = list(generated)
    result.sort()

    if len(result) != n_seeds:
        raise ValueError("Failed to generate {n_seeds} unique seeds.")
    return result


@numba.njit(nogil=True)
def is_all_finite(a: numpy.ndarray) -> bool:
    """
    Check if an array is all finite.

    :param a: the input array
    :return: `True` if all elements in the array are finite, `False` otherwise

    >>> import numpy as npx
    >>> from moptipy.utils.nputils import is_all_finite
    >>> print(is_all_finite(npx.array([1.1, 2.1, 3])))
    True
    >>> print(is_all_finite(npx.array([1, 2, 3])))
    True
    >>> print(is_all_finite(npx.array([1.1, npx.inf, 3])))
    False
    """
    for x in a:
        if not numpy.isfinite(x):
            return False
    return True


#: the character identifying the numpy data type backing the space
KEY_NUMPY_TYPE: Final[str] = "dtype"


def val_numpy_type(dtype: numpy.dtype) -> str:
    """
    Convert a numpy data type to a string.

    :param dtype: the data type
    :returns: a string representation

    >>> import numpy as npx
    >>> val_numpy_type(npx.dtype(int))
    'l'
    >>> val_numpy_type(npx.dtype(float))
    'd'
    """
    return dtype.char


def np_to_py_number(number) -> Union[int, float]:
    """
    Convert a scalar number from numpy to a corresponding Python type.

    :param number: the numpy number
    :returns: an integer or float representing the number
    """
    if isinstance(number, (int, float)):
        return number
    if isinstance(number, numpy.number):
        if isinstance(number, numpy.integer):
            return int(number)
        if isinstance(number, numpy.floating):
            return float(number)
    raise type_error(number, "number",
                     (int, float, numpy.integer, numpy.floating))


def array_to_str(data: numpy.ndarray) -> str:
    """
    Convert a numpy array to a string.

    This method represents a numpy array as a string.
    It makes sure to include all the information stored in the array and to
    represent it as compactly as possible.

    If the array has numerical values, it will use the default CSV separator
    (:const:`~moptipy.utils.logger.CSV_SEPARATOR`).
    If the array contains Boolean values, it will use no separator at all.

    :param data: the data
    :returns: the string

    >>> import numpy as npx
    >>> array_to_str(npx.array([1, 2, 3]))
    '1;2;3'
    >>> array_to_str(npx.array([1, 2.2, 3]))
    '1;2.2;3'
    >>> array_to_str(npx.array([True, False, True]))
    'TFT'
    """
    if not isinstance(data, numpy.ndarray):
        raise type_error(data, "data", numpy.ndarray)
    k: Final[str] = data.dtype.kind
    if k in ('i', 'u'):
        return CSV_SEPARATOR.join(str(d) for d in data)
    if k == 'f':
        return CSV_SEPARATOR.join(num_to_str(float(d)) for d in data)
    if k == 'b':
        return "".join(bool_to_str(bool(d)) for d in data)
    raise ValueError(f"unsupported data kind '{k}' of type '{data.dtype}'.")
