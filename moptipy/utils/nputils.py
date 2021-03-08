"""Utilities for interaction with numpy."""
from hashlib import sha512
from typing import Final, Optional, List, Iterable
from typing import Set, Dict, Tuple, cast

import numpy as np
from numpy.random import default_rng, Generator, PCG64

from moptipy.utils import logging

__NP_INTS: Final[Tuple[Tuple[np.dtype, int, int], ...]] = \
    tuple([(t[0], int(t[1].min), int(t[1].max))
           for t in ((tt, np.iinfo(tt))
                     for tt in (cast(np.dtype, np.dtype(ttt))
                                for ttt in [np.int8, np.uint8,
                                            np.int16, np.uint16,
                                            np.int32, np.uint32,
                                            np.int64, np.uint64]))])

#: The default integer type: the signed 64 bit integer
DEFAULT_INT: Final[np.dtype] = (__NP_INTS[len(__NP_INTS) - 2])[0]

#: The default unsigned integer type: an unsigned 64 bit integer
__DEFAULT_UNSIGNED_INT: Final[np.dtype] = (__NP_INTS[len(__NP_INTS) - 1])[0]

__NP_INT_MAX: Final[Dict[np.dtype, int]] = {}
for i in __NP_INTS:
    __NP_INT_MAX[i[0]] = i[2]
__NP_INT_MIN: Final[Dict[np.dtype, int]] = {}
for i in __NP_INTS:
    __NP_INT_MIN[i[0]] = i[1]


def int_range_to_dtype(min_value: int, max_value: int) -> np.dtype:
    """
    Convert an integer range to an appropriate numpy data type.

    The returned type is as compact as possible and signed types are
    preferred over unsigned types.
    The returned :class:`np.dtype` will allow accommodating all values in the
    inclusive interval `min_value..max_value`.

    :param int min_value: the minimum value
    :param int max_value: the maximum value
    :return: the numpy integer range
    :rtype: np.dtype
    :raises TypeError: if the parameters are not integers
    :raises ValueError: if the range is invalid
    """
    if not isinstance(min_value, int):
        raise TypeError("min_value must be int, but is '"
                        + str(type(min_value)) + "'.")
    if not isinstance(max_value, int):
        raise TypeError("max_value must be int, but is '"
                        + str(type(max_value)) + "'.")
    if min_value > max_value:
        raise ValueError("min_value must be <>>= max_value, but min_value="
                         + str(min_value) + " and max_value="
                         + str(max_value) + " was provided.")

    for t in __NP_INTS:
        if (min_value >= t[1]) and (max_value <= t[2]):
            return t[0]

    ll = len(__NP_INTS) - 1
    if min_value >= 0:
        raise ValueError("max_value for unsigned integers must be "
                         "<=" + str((__NP_INTS[ll])[2]) + ", but is "
                         + str(max_value) + " for min_value "
                         + str(min_value) + ".")

    ll = ll - 1
    raise ValueError("Signed integer range cannot exceed "
                     + str((__NP_INTS[ll])[1]) + ".." + str((__NP_INTS[ll])[2])
                     + ", but " + str(min_value) + ".." + str(max_value)
                     + " was specified.")


def intmax(shape, dtype: np.dtype = DEFAULT_INT) -> np.ndarray:
    """
    Create an integer array of the given length filled with the maximum value.

    :param shape: the requested shape
    :param dtype: the data type (defaults to 64 bit integers)
    :return: the new array
    :rtype: np.ndarray
    """
    return np.full(shape=shape,
                   fill_value=__NP_INT_MAX[dtype],
                   dtype=dtype)


def intmin(shape, dtype: np.dtype = DEFAULT_INT) -> np.ndarray:
    """
    Create an integer array of the given length filled with the minimum value.

    :param shape: the requested shape
    :param dtype: the data type (defaults to 64 bit integers)
    :return: the new array
    :rtype: np.ndarray
    """
    return np.full(shape=shape,
                   fill_value=__NP_INT_MIN[dtype],
                   dtype=dtype)


__SEED_BYTES: Final = 8
__MIN_RAND_SEED: Final = 0
__MAX_RAND_SEED: Final = int((1 << (__SEED_BYTES * 8)) - 1)


def rand_seed_generate(random: Optional[Generator] = None) -> int:
    """
    Generate a random seed.

    :param Optional[Generator] random: the random number generator to be used
        to generate the seed
    :return: the random seed
    :rtype: int
    :raises TypeError: if `random` is specified but is not an instance of
        `Generator`
    """
    if random is None:
        random = default_rng()
    if not isinstance(random, Generator):
        raise TypeError("random must be instance of Generator, but is "
                        + str(type(random)) + ".")
    return int.from_bytes(random.bytes(__SEED_BYTES),
                          byteorder='big', signed=False)


def rand_seed_check(rand_seed: int) -> int:
    """
    Make sure that a random seed is valid.

    :param int rand_seed: the random seed to check
    :return: the rand seed
    :rtype: int

    :raises TypeError: if the random seed is not an `int`
    :raises ValueError: if the random seed is not valid
    """
    if not isinstance(rand_seed, int):
        raise TypeError("rand_seed should be instance of int, but is "
                        + str(type(rand_seed)) + ".")
    if (rand_seed < __MIN_RAND_SEED) or (rand_seed > __MAX_RAND_SEED):
        raise ValueError("rand_seed must be in " + str(__MIN_RAND_SEED)
                         + ".." + str(__MAX_RAND_SEED) + ", but is "
                         + str(rand_seed) + ".")
    return rand_seed


def rand_generator(seed: int) -> Generator:
    """
    Instantiate a random number generator from a seed.

    :param int seed: the random seed
    :return: the random number generator
    :rtype: Generator
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

    :param str string: the string
    :param int n_seeds: the number of seeds
    :return: a tuple of random seeds
    :rtype: List[int]
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
        raise TypeError(
            "string must be a str, but is '" + str(type(string)) + "'.")
    if len(string) <= 0:
        raise ValueError("string must not be empty.")
    if not isinstance(n_seeds, int):
        raise TypeError(
            "n_seeds must be an int, but is '" + str(type(string)) + "'.")
    if n_seeds <= 0:
        raise ValueError(
            "n_seeds must be positive, but is " + str(n_seeds) + ".")

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
        raise ValueError("Failed to generate " + str(n_seeds)
                         + " unique seeds.")
    return result


#: The default boolean type
__DEFAULT_BOOLS: Final[np.dtype] = np.dtype(np.bool_)


def strs_to_bools(lines: Iterable[str]) -> np.ndarray:
    """
    Convert an array of strings to a boolean numpy array.

    :param Iterable[str] lines: the lines
    :return: the array
    :rtype: np.ndarray

    >>> strs_to_bools(["T", "F", "T"])
    array([ True, False,  True])
    """
    return np.array([logging.str_to_bool(s) for s in lines],
                    dtype=__DEFAULT_BOOLS)


def strs_to_uints(lines: Iterable[str]) -> np.ndarray:
    """
    Convert an array of strings to a numpy array of unsigned ints.

    :param Iterable[str] lines: the lines
    :return: the array
    :rtype: np.ndarray

    >>> strs_to_uints(["1", "2", "3"])
    array([1, 2, 3], dtype=uint64)
    """
    return np.array(lines, dtype=__DEFAULT_UNSIGNED_INT)


def strs_to_ints(lines: Iterable[str]) -> np.ndarray:
    """
    Convert an array of strings to a numpy array of signed ints.

    :param Iterable[str] lines: the lines
    :return: the array
    :rtype: np.ndarray

    >>> strs_to_ints(["-1", "2", "3"])
    array([-1,  2,  3])
    """
    return np.array(lines, dtype=DEFAULT_INT)


#: The default float type
__DEFAULT_FLOAT: Final[np.dtype] = np.dtype(np.float64)


def strs_to_floats(lines: Iterable[str]) -> np.ndarray:
    """
    Convert an array of strings to a numpy array of floats.

    :param Iterable[str] lines: the lines
    :return: the array
    :rtype: np.ndarray

    >>> strs_to_floats(["-1.6", "2", "3"])
    array([-1.6,  2. ,  3. ])
    """
    return np.array(lines, dtype=__DEFAULT_FLOAT)
