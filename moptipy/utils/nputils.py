from hashlib import sha512
from typing import Final, Optional, Tuple

import numpy as np
from numpy.random import default_rng, Generator, PCG64

__NP_INTS: Final = \
    tuple([(t[0], int(t[1].min), int(t[1].max))
           for t in ((tt, np.iinfo(tt))
                     for tt in (np.dtype(ttt)
                                for ttt in [np.int8, np.uint8,
                                            np.int16, np.uint16,
                                            np.int32, np.uint32,
                                            np.int64, np.uint64]))])

DEFAULT_INT: Final = (__NP_INTS[len(__NP_INTS) - 2])[0]

__NP_INT_MAX: Final = {}
for i in __NP_INTS:
    __NP_INT_MAX[i[0]] = i[2]
__NP_INT_MIN: Final = {}
for i in __NP_INTS:
    __NP_INT_MIN[i[0]] = i[1]


def int_range_to_dtype(min_value: int, max_value: int) -> np.dtype:
    """
    Convert an integer range to an appropriate numpy data type.
    The returned type is as compact as possible and signed types are
    preferred over unsigned types.
    The returned dtype will allow accommodating all values in the
    inclusive interval min_value..max_value.

    :param int min_value: the minimum value
    :param int max_value: the maximum value
    :return np.dtype: the numpy integer range
    """

    if not isinstance(min_value, int):
        raise ValueError("min_value must be int, but is '"
                         + str(type(min_value)) + "'.")
    if not isinstance(max_value, int):
        raise ValueError("max_value must be int, but is '"
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
    Create an integer array of the given length filled with the maximum value
    :param shape: the requested shape
    :param dtype: the data type (defaults to 64 bit integers)
    :return: the new array
    """
    return np.full(shape=shape,
                   fill_value=__NP_INT_MAX[dtype],
                   dtype=dtype)


def intmin(shape, dtype: np.dtype = DEFAULT_INT) -> np.ndarray:
    """
    Create an integer array of the given length filled with the minimum value
    :param shape: the requested shape
    :param dtype: the data type (defaults to 64 bit integers)
    :return: the new array
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
    """
    if random is None:
        random = default_rng()
    if not isinstance(random, Generator):
        raise ValueError("random must be instance of Generator, but is "
                         + str(type(random)) + ".")
    return int.from_bytes(random.bytes(__SEED_BYTES),
                          byteorder='big', signed=False)


def rand_seed_check(rand_seed: int) -> int:
    """
    Make sure that a random seed is valid.
    :param int rand_seed: the random seed to check
    :return: the rand seed
    :rtype: int
    :raises ValueError: if the random seed is not valid
    """
    if not isinstance(rand_seed, int):
        raise ValueError("rand_seed should be instance of int, but is "
                         + str(type(rand_seed)) + ".")
    if (rand_seed < __MIN_RAND_SEED) or (rand_seed > __MAX_RAND_SEED):
        raise ValueError("rand_seed must be in " + str(__MIN_RAND_SEED)
                         + ".." + str(__MAX_RAND_SEED) + ", but is "
                         + str(rand_seed) + ".")
    return rand_seed


def rand_generator(seed: int) -> Generator:
    """
    Instantiate a random number generator from a seed
    :param int seed: the random seed
    :return: the random number generator
    :rtype: Generator
    """
    return default_rng(rand_seed_check(seed))


def rand_seeds_from_str(string: str,
                        n_seeds: int) -> Tuple[int]:
    """
    In a reproducible fashion, generate `n_seeds` unique random number
    seeds from a `string`
    :param str string: the string
    :param int n_seeds: the number of seeds
    :return: a tuple of random seeds
    :rtype: Tuple[int]
    """
    if not isinstance(string, str):
        raise ValueError(
            "string must be a str, but is '" + str(type(string)) + "'.")
    if len(string) <= 0:
        raise ValueError("string must not be empty.")
    if not isinstance(n_seeds, int):
        raise ValueError(
            "n_seeds must be an int, but is '" + str(type(string)) + "'.")
    if n_seeds <= 0:
        raise ValueError(
            "n_seeds must be positive, but is " + str(n_seeds) + ".")

    seeds = bytearray(sha512(string.encode("utf8")).digest())
    seeds = [int.from_bytes(seeds[ii:(ii + 16)],
                            byteorder='big', signed=False)
             for ii in range(0, len(seeds), 16)]
    if len(seeds) != 4:
        raise ValueError("Did not produce 4 numbers of 128 bit from string?")

    # seed two PCG64 generators, each of which should use two 128 bit
    # numbers as seed
    g1 = Generator(PCG64(seeds[0:2]))
    g2 = Generator(PCG64(seeds[2:4]))

    generated = set()
    while len(generated) < n_seeds:
        a = g1
        g1 = g2
        g2 = a
        generated.add(rand_seed_generate(g1))

    result = list(generated)
    result.sort()
    result = tuple(result)

    if len(result) != n_seeds:
        raise ValueError("Failed to generate " + str(n_seeds)
                         + " unique seeds.")
    return result
