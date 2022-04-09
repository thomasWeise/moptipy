"""Utilities for interaction with numpy."""
from hashlib import sha512
from typing import Final, List, Iterable, cast
from typing import Set, Dict, Tuple

import numba  # type: ignore
import numpy
from numpy.random import default_rng, Generator, PCG64

from moptipy.utils.strings import str_to_bool
from moptipy.utils.types import type_error

#: A map associating all numpy integer types associated to tuples
#: of their respective minimum and maximum value.
__NP_INTS_MAP: Final[Dict[numpy.dtype, Tuple[int, int]]] = \
    {t: (int(ti.min), int(ti.max)) for t, ti in
     cast(List[Tuple[numpy.dtype, numpy.iinfo]],
          ((dtt, numpy.iinfo(dtt)) for dtt in
           cast(List[numpy.dtype], [numpy.dtype(dt) for dt in
                                    cast(List[type],
                                         [numpy.int8, numpy.uint8,
                                          numpy.int16, numpy.uint16,
                                          numpy.int32, numpy.uint32,
                                          numpy.int64, numpy.uint64])])))}

#: A tuple of integer types with their corresponding minimum and maximum
#: values in increasing orders of size. The tuple contains alternating
#: signed and unsigned types. It starts with the smallest signed type,
#: `numpy.int8` and ends with the largest unsigned type `numpy.uint64`.
#: If we have a range `[min..max]` of valids values, then we can look up this
#: range and find the integer type with the smallest memory footprint to
#: accommodate it. This is what :meth:`int_range_to_dtype` does.
__NP_INTS_LIST: Final[Tuple[Tuple[numpy.dtype, int, int], ...]] = \
    tuple([(a[3], a[1], a[2]) for a in
           sorted([(t.itemsize, a[0], a[1], t)
                   for t, a in __NP_INTS_MAP.items()])])

#: The default integer type: the signed 64 bit integer.
DEFAULT_INT: Final[numpy.dtype] = (__NP_INTS_LIST[-2])[0]

#: The default unsigned integer type: an unsigned 64 bit integer.
__DEFAULT_UNSIGNED_INT: Final[numpy.dtype] = (__NP_INTS_LIST[-1])[0]

#: The default boolean type.
DEFAULT_BOOL: Final[numpy.dtype] = numpy.dtype(numpy.bool_)

#: The default floating point type.
DEFAULT_FLOAT: Final[numpy.dtype] = numpy.zeros(1).dtype


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

    for t in __NP_INTS_LIST:
        if (min_value >= t[1]) and (max_value <= t[2]):
            return t[0]

    if min_value >= 0:
        raise ValueError(
            "max_value for unsigned integers must be <="
            f"{(__NP_INTS_LIST[-1])[2]}, but is {max_value} "
            f" for min_value={min_value}.")

    raise ValueError(
        f"Signed integer range cannot exceed {__NP_INTS_LIST[-2][1]}.."
        f"{__NP_INTS_LIST[-2][2]}, but {min_value}..{max_value} "
        "was specified.")


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


@numba.jit(forceobj=True)
def np_ints_min(shape, dtype: numpy.dtype = DEFAULT_INT) -> numpy.ndarray:
    """
    Create an integer array of the given length filled with the minimum value.

    :param shape: the requested shape
    :param dtype: the data type (defaults to 64 bit integers)
    :return: the new array

    >>> import numpy as npx
    >>> from moptipy.utils.nputils import np_ints_min
    >>> print(np_ints_min(3, npx.dtype("int16")))
    [-32768 -32768 -32768]
    """
    return numpy.full(shape=shape,
                      fill_value=__NP_INTS_MAP[dtype][0],
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


def strs_to_bools(lines: Iterable[str]) -> numpy.ndarray:
    """
    Convert an array of strings to a boolean numpy array.

    :param lines: the lines
    :return: the array

    >>> strs_to_bools(["T", "F", "T"])
    array([ True, False,  True])
    """
    return numpy.array([str_to_bool(s) for s in lines],
                       dtype=DEFAULT_BOOL)


def strs_to_uints(lines: Iterable[str]) -> numpy.ndarray:
    """
    Convert an array of strings to a numpy array of unsigned ints.

    :param lines: the lines
    :return: the array

    >>> strs_to_uints(["1", "2", "3"])
    array([1, 2, 3], dtype=uint64)
    """
    return numpy.array(lines, dtype=__DEFAULT_UNSIGNED_INT)


def strs_to_ints(lines: Iterable[str]) -> numpy.ndarray:
    """
    Convert an array of strings to a numpy array of signed ints.

    :param lines: the lines
    :return: the array

    >>> strs_to_ints(["-1", "2", "3"])
    array([-1,  2,  3])
    """
    return numpy.array(lines, dtype=DEFAULT_INT)


def strs_to_floats(lines: Iterable[str]) -> numpy.ndarray:
    """
    Convert an array of strings to a numpy array of floats.

    :param lines: the lines
    :return: the array

    >>> strs_to_floats(["-1.6", "2", "3"])
    array([-1.6,  2. ,  3. ])
    """
    return numpy.array(lines, dtype=DEFAULT_FLOAT)


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
