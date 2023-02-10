"""
A Python implementation of the W-Model benchmark problem.

This is a tunable benchmark problem for bit-string based search spaces.
It exhibits ruggedness, epistasis, deceptiveness, and (uniform) neutrality in
a tunable fashion.
In [1], a set of 19 diverse instances of the W-Model are selected for
algorithm benchmarking. These instances are provided via
:meth:`~moptipy.examples.bitstrings.w_model.WModel.default_instances`.

1. Thomas Weise, Yan Chen, Xinlu Li, and Zhize Wu. Selecting a diverse set of
   benchmark instances from a tunable model problem for black-box discrete
   optimization algorithms. *Applied Soft Computing Journal (ASOC)*, 92:106269,
   June 2020. doi: https://doi.org/10.1016/j.asoc.2020.106269
2. Thomas Weise and Zijun Wu. Difficult Features of Combinatorial Optimization
   Problems and the Tunable W-Model Benchmark Problem for Simulating them. In
   *Black Box Discrete Optimization Benchmarking (BB-DOB) Workshop* of
   *Companion Material Proceedings of the Genetic and Evolutionary Computation
   Conference (GECCO 2018),* July 15th-19th 2018, Kyoto, Japan,
   pages 1769-1776, ISBN 978-1-4503-5764-7. ACM.
   doi: https://doi.org/10.1145/3205651.3208240
3. Thomas Weise, Stefan Niemczyk, Hendrik Skubch, Roland Reichle, and
   Kurt Geihs. A Tunable Model for Multi-Objective, Epistatic, Rugged, and
   Neutral Fitness Landscapes. In Maarten Keijzer, Giuliano Antoniol,
   Clare Bates Congdon, Kalyanmoy Deb, Benjamin Doerr, Nikolaus Hansen,
   John H. Holmes, Gregory S. Hornby, Daniel Howard, James Kennedy,
   Sanjeev P. Kumar, Fernando G. Lobo, Julian Francis Miller, Jason H. Moore,
   Frank Neumann, Martin Pelikan, Jordan B. Pollack, Kumara Sastry,
   Kenneth Owen Stanley, Adrian Stoica, El-Ghazali, and Ingo Wegener, editors,
   *Proceedings of the 10th Annual Conference on Genetic and Evolutionary
   Computation (GECCO'08)*, pages 795-802, July 12-16, 2008, Atlanta, GA, USA.
   ISBN: 978-1-60558-130-9, New York, NY, USA: ACM Press.
   doi: https://doi.org/10.1145/1389095.1389252
4. Carola Doerr, Furong Ye, Naama Horesh, Hao Wang, Ofer M. Shir, Thomas Bäck.
   Benchmarking Discrete Optimization Heuristics with IOHprofiler. *Applied
   Soft Computing* 88:106027, March 2020,
   doi: https://doi.org/10.1016/j.asoc.2019.106027.
"""

from array import array
from math import sqrt
from typing import Callable, Final, Iterable, cast

import numba  # type: ignore
import numpy as np

from moptipy.examples.bitstrings.bitstring_problem import BitStringProblem
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_BOOL
from moptipy.utils.strings import sanitize_name
from moptipy.utils.types import check_int_range, type_error


@numba.njit(nogil=True, cache=True, inline="always")
def w_model_f(x: np.ndarray) -> int:
    """
    Compute the basic hamming distance of the W-Model to the optimal string.

    The optimal string in the W-Model is `0101010101...`. Here we compute the
    objective value of a candidate solution, i.e., the Hamming distance to
    the string `010101010101...`. This is the basic problem objective
    function. It can be applied either directly, on top of transformations
    such as the neutrality- or epistasis mappings. Its result can be
    transformed by a ruggedness permutation to introduce ruggedness into the
    problem.

    :param x: the bit string to evaluate
    :return: the Hamming distance to the string of alternating zeros and ones
        and starting with `0`.

    >>> w_model_f(np.array([False]))
    0
    >>> w_model_f(np.array([True]))
    1
    >>> w_model_f(np.array([False, True]))
    0
    >>> w_model_f(np.array([False, False]))
    1
    >>> w_model_f(np.array([True, True]))
    1
    >>> w_model_f(np.array([True, False]))
    2
    >>> w_model_f(np.array([False, True, False]))
    0
    >>> w_model_f(np.array([True, False, True]))
    3
    >>> w_model_f(np.array([True, True, True]))
    2
    >>> w_model_f(np.array([True, True, False]))
    1
    >>> w_model_f(np.array([False, True, True]))
    1
    >>> w_model_f(np.array([False, True, False, True, False, True, False]))
    0
    >>> w_model_f(np.array([False, True, False, True, False, True, True]))
    1
    >>> w_model_f(np.array([True, False, True, False, True, False, True]))
    7
    """
    result = 0
    for i, xx in enumerate(x):
        if xx == ((i & 1) == 0):
            result = result + 1
    return result


@numba.njit(nogil=True, cache=True, inline="always")
def w_model_neutrality(x_in: np.ndarray, mu: int, x_out: np.ndarray) -> None:
    """
    Perform the neutrality transformation.

    The neutrality transformation is the first layer of the W-Model. It
    introduces (or better, removes during the mapping) uniform redundancy by
    basically reducing the size of a bit string by factor `mu`.

    Basically, the input array `x_in` is split in consecutive bit groups of
    length `mu`. Each such group is translated to one bit in the output string
    `x_out`. This bit will become `1` if the majority of the bits in the input
    group are also `1`. Otherwise it becomes `0`.

    Notice that `len(x_out) >= mu * length(x_in)` must hold.

    :param x_in: the input array
    :param mu: the number of bits to merge
    :param x_out: the output array

    >>> xin = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
    ...                 0, 1, 1, 1, 0, 1, 0, 0, 0, 0], bool)
    >>> xout = np.empty(len(xin) // 2, bool)
    >>> w_model_neutrality(xin, 2, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    1011001110
    >>> xin = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0,
    ...                 0, 1, 1, 1, 0, 1, 0, 0, 0, 0], bool)
    >>> w_model_neutrality(xin, 2, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    1111001110
    """
    threshold_for_1: Final[int] = (mu >> 1) + (mu & 1)
    flush: int = mu
    i_in: int = 0
    i_out: int = 0
    ones: int = 0
    length_in: Final[int] = len(x_in)
    length_out: Final[int] = len(x_out)
    while (i_in < length_in) and (i_out < length_out):
        if x_in[i_in]:
            ones = ones + 1
        i_in = i_in + 1
        if i_in >= flush:
            flush = flush + mu
            x_out[i_out] = (ones >= threshold_for_1)
            i_out = i_out + 1
            ones = 0


@numba.njit(nogil=True, cache=True, inline="always")
def __w_model_epistasis(x_in: np.ndarray, start: int,
                        nu: int, x_out: np.ndarray) -> None:
    """
    Perform the transformation of an epistasis chunk.

    :param x_in: the input string
    :param start: the start index
    :param nu: the epistasis parameter
    :param x_out: the output string

    >>> xout = np.empty(4, bool)
    >>> __w_model_epistasis(np.array([0, 0, 0, 0], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    0000
    >>> __w_model_epistasis(np.array([0, 0, 0, 1], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    1101
    >>> __w_model_epistasis(np.array([0, 0, 1, 0], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    1011
    >>> __w_model_epistasis(np.array([0, 1, 0, 0], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    0111
    >>> __w_model_epistasis(np.array([1, 0, 0, 0], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    1111
    >>> __w_model_epistasis(np.array([1, 1, 1, 1], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    1110
    >>> __w_model_epistasis(np.array([0, 1, 1, 1], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    0001
    >>> __w_model_epistasis(np.array([1, 0, 1, 1], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    1001
    >>> __w_model_epistasis(np.array([1, 1, 0, 1], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    0101
    >>> __w_model_epistasis(np.array([1, 1, 1, 0], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    0011
    >>> __w_model_epistasis(np.array([0, 0, 1, 1], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    0110
    >>> __w_model_epistasis(np.array([0, 1, 0, 1], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    1010
    >>> __w_model_epistasis(np.array([0, 1, 1, 0], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    1100
    >>> __w_model_epistasis(np.array([1, 0, 0, 1], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    0010
    >>> __w_model_epistasis(np.array([1, 0, 1, 0], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    0100
    >>> __w_model_epistasis(np.array([1, 1, 0, 0], bool), 0, 4, xout)
    >>> print("".join('1' if xx else '0' for xx in xout))
    1000
    """
    end: Final[int] = (start + nu) - 1
    flip: Final[bool] = x_in[start]
    i: int = end
    skip: int = start
    while i >= start:
        result: bool = flip
        j: int = end
        while j > start:
            if j != skip:
                result = result ^ x_in[j]
            j = j - 1
        x_out[i] = result
        skip = skip - 1
        if skip < start:
            skip = end
        i = i - 1


@numba.njit(nogil=True, cache=True, inline="always")
def w_model_epistasis(x_in: np.ndarray, nu: int, x_out: np.ndarray) -> None:
    """
    Perform the epistasis transformation.

    :param x_in: the input string
    :param nu: the epistasis parameter
    :param x_out: the output string
    """
    total_len: Final[int] = len(x_in)
    end: Final[int] = total_len - nu
    i: int = 0
    while i <= end:
        __w_model_epistasis(x_in, i, nu, x_out)
        i = i + nu
    if i < total_len:
        __w_model_epistasis(x_in, i, total_len - i, x_out)


@numba.njit(nogil=True, cache=True, inline="always")
def w_model_max_gamma(nopt: int) -> int:
    """
    Compute the maximum gamma value for a given length `nopt`.

    :param nopt: the length of the optimal bit string
    :return: the maximum gamma value
    """
    return int((nopt * (nopt - 1)) >> 1)


@numba.njit(nogil=False, cache=True, inline="always")
def w_model_create_ruggedness_permutation(
        gamma: int, nopt: int, r: array) -> None:
    """
    Create the raw ruggedness array.

    This function creates the ruggedness permutation where groups of rugged
    transformations alternate with deceptive permutations for increasing
    gamma.

    :param gamma: the parameter for the ruggedness
    :param nopt: the length of the optimal string
    :param r: the array to receive the permutation

    >>> r = array('i', range(11))
    >>> w_model_create_ruggedness_permutation(0, 10, r)
    >>> "".join(str(xx) for xx in r)
    '012345678910'
    >>> r = array('i', range(7))
    >>> w_model_create_ruggedness_permutation(9, 6, r)
    >>> r[3]
    3
    >>> r[6]
    5
    """
    max_gamma: Final[int] = w_model_max_gamma(nopt)
    start: int = 0 if gamma <= 0 else (
        nopt - 1 - int(0.5 + sqrt(0.25 + ((max_gamma - gamma) << 1))))
    k: int = 0
    j: int = 1
    while j <= start:
        if (j & 1) != 0:
            r[j] = nopt - k
        else:
            k = k + 1
            r[j] = k
        j = j + 1

    while j <= nopt:
        k = k + 1
        r[j] = (nopt - k) if ((start & 1) != 0) else k
        j = j + 1

    upper: Final[int] = ((gamma - max_gamma) + (
        ((nopt - start - 1) * (nopt - start)) >> 1))
    j = j - 1
    i: int = 1
    while i <= upper:
        j = j - 1
        r[j], r[nopt] = r[nopt], r[j]
        i = i + 1


@numba.njit(nogil=True, cache=True, inline="always")
def w_model_ruggedness_translate(gamma: int, nopt: int) -> int:
    """
    Transform the gamma values such that deceptive follows rugged.

    If the ruggedness permutations are created with raw gamma values, then
    rugged and deceptive permutations will alternate with rising gamma.
    With this function here, the gamma parameters are translated such that
    first all the rugged permutations come (with rising degree of ruggedness)
    and then the deceptive ones come.

    :param gamma: the parameter for the ruggedness
    :param nopt: the length of the optimal string
    :returns: the translated gamma values

    >>> w_model_ruggedness_translate(12, 6)
    9
    >>> w_model_ruggedness_translate(34, 25)
    57
    >>> w_model_ruggedness_translate(0, 5)
    0
    >>> w_model_ruggedness_translate(1, 5)
    1
    >>> w_model_ruggedness_translate(2, 5)
    2
    >>> w_model_ruggedness_translate(3, 5)
    3
    >>> w_model_ruggedness_translate(4, 5)
    4
    >>> w_model_ruggedness_translate(5, 5)
    8
    >>> w_model_ruggedness_translate(6, 5)
    9
    >>> w_model_ruggedness_translate(7, 5)
    10
    >>> w_model_ruggedness_translate(8, 5)
    7
    >>> w_model_ruggedness_translate(9, 5)
    6
    >>> w_model_ruggedness_translate(10, 5)
    5
    """
    if gamma <= 0:
        return 0

    last_upper: int = (nopt >> 1) * ((nopt + 1) >> 1)
    if gamma <= last_upper:
        j = int(((nopt + 2) * 0.5) - sqrt(
            (((nopt * nopt) * 0.25) + 1) - gamma))
        k = ((gamma - ((nopt + 2) * j)) + (j * j) + nopt)
        return int((k + 1 + ((((nopt + 2) * j) - (j * j) - nopt - 1) << 1))
                   - (j - 1))

    j = int((((nopt % 2) + 1) * 0.5) + sqrt(
        (((1 - (nopt % 2)) * 0.25) + gamma) - 1 - last_upper))
    k = gamma - (((j - (nopt % 2)) * (j - 1)) + 1 + last_upper)
    return int(w_model_max_gamma(nopt) - k - ((2 * j * j) - j)
               - ((nopt % 2) * ((-2 * j) + 1)))


@numba.njit(nogil=True, cache=True, inline="always")
def _w_model(x_in: np.ndarray, m: int, m_out: np.ndarray | None,
             nu: int, nu_out: np.ndarray | None, r: array | None) -> int:
    """
    Compute the value of the multi-stage W-model.

    :param x_in: the input array
    :param m: the neutrality level
    :param m_out: the temporary variable for the de-neutralized `x_in`
    :param nu: the epistasis level
    :param nu_out: the temporary variable for the epistasis transformed string
    :param r: the ruggedness transform, if any
    :returns: the objective value
    """
    neutral: np.ndarray
    if m_out is None:
        neutral = x_in
    else:
        neutral = m_out
        w_model_neutrality(x_in, m, neutral)
    epi: np.ndarray
    if nu_out is None:
        epi = neutral
    else:
        epi = nu_out
        w_model_epistasis(neutral, nu, epi)
    f = w_model_f(epi)
    if r is None:
        return int(f)
    return r[f]


class WModel(BitStringProblem):
    """The tunable W-Model benchmark problem."""

    def __init__(self, nopt: int, m: int = 1, nu: int = 2,
                 gamma: int = 0, name: str | None = None) -> None:
        """
        Initialize an W-Model instance.

        :param nopt: the length of the optimal string
        :param m: the neutrality parameter
        :param nu: the epistasis parameter
        :param gamma: the ruggedness parameter
        :param name: the (optional) special name of this instance
        """
        super().__init__(nopt * m)
        #: the length of the optimal string
        self.nopt: Final[int] = check_int_range(nopt, "nopt", 2)
        #: the neutrality parameter
        self.m: Final[int] = check_int_range(m, "m", 1)
        #: the internal buffer for de-neutralization
        self.__m_out: Final[np.ndarray | None] = None if m < 2 \
            else np.empty(nopt, DEFAULT_BOOL)
        #: the epistasis parameter
        self.nu: Final[int] = check_int_range(nu, "nu", 2, nopt)
        #: the normalized epistasis parameter
        self.nu1: Final[float] = (nu - 2) / (nopt - 2)
        #: the internal buffer for de-epistazation
        self.__nu_out: Final[np.ndarray | None] = None if nu <= 2 \
            else np.empty(nopt, DEFAULT_BOOL)
        max_gamma: Final[int] = w_model_max_gamma(nopt)
        #: the ruggedness parameter
        self.gamma: Final[int] = check_int_range(gamma, "gamma", 0, max_gamma)
        #: the normalized ruggedness parameter
        self.gamma1: Final[float] = gamma / max_gamma
        #: the translated gamma parameter
        self.gamma_prime: Final[int] = w_model_ruggedness_translate(
            gamma, nopt)
        r: array | None = None  # the ruggedness table
        if gamma > 0:
            r = array("i", range(nopt + 1))
            w_model_create_ruggedness_permutation(self.gamma_prime, nopt, r)
        #: the ruggedness table
        self.__r: Final[array | None] = r

        if name is None:
            name = f"wmodel_{self.nopt}_{self.m}_{self.nu}_{self.gamma}"
        else:
            if not isinstance(name, str):
                raise type_error(name, "name", str)
            if (len(name) <= 0) or (sanitize_name(name) != name):
                raise ValueError(f"invalid name: {name!r}")
        #: the name of this w-model instance
        self.name: Final[str] = name

    def upper_bound(self) -> int:
        """
        Get the upper bound of the bit string based problem.

        :return: the length of the bit string

        >>> print(WModel(7, 6, 4, 4).upper_bound())
        7
        """
        return self.nopt

    def evaluate(self, x: np.ndarray) -> int:
        """
        Evaluate a solution to the W-Model.

        :param x: the bit string to evaluate
        :returns: the value of the W-Model for the string
        """
        return _w_model(x, self.m, self.__m_out, self.nu,
                        self.__nu_out, self.__r)

    def __str__(self):
        """
        Get the name of the W-Model instance.

        :returns: the name of the W-Model instance.
        """
        return self.name

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         WModel(6, 2, 4, 7).log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[1]
        'name: wmodel_6_2_4_7'
        >>> text[3]
        'lowerBound: 0'
        >>> text[4]
        'upperBound: 6'
        >>> text[5]
        'n: 12'
        >>> text[6]
        'nopt: 6'
        >>> text[7]
        'm: 2'
        >>> text[8]
        'nu: 4'
        >>> text[9]
        'nu1: 0.5'
        >>> text[11]
        'gamma: 7'
        >>> text[12]
        'gamma1: 0.4666666666666667'
        >>> text[14]
        'gammaPrime: 11'
        >>> len(text)
        16
        """
        super().log_parameters_to(logger)
        logger.key_value("nopt", self.nopt)
        logger.key_value("m", self.m)
        logger.key_value("nu", self.nu)
        logger.key_value("nu1", self.nu1)
        logger.key_value("gamma", self.gamma)
        logger.key_value("gamma1", self.gamma1)
        logger.key_value("gammaPrime", self.gamma_prime)

    @staticmethod
    def default_instances() -> Iterable[Callable[[], "WModel"]]:
        """
        Get the 19 default instances of the W-Model.

        :returns: an `Iterable` that can provide callables constructing the
            19 default instances of the W-Model

        >>> len(list(WModel.default_instances()))
        19
        """
        return (cast(Callable[[], "WModel"],
                     lambda a=iid, b=z[0], c=z[1], d=z[2], g=z[3]:
                     WModel(b, c, d, g, f"wmodel{a + 1}"))
                for iid, z in enumerate([
                    (10, 2, 6, 10), (10, 2, 6, 18), (16, 1, 5, 72),
                    (16, 3, 9, 72), (25, 1, 23, 90), (32, 1, 2, 397),
                    (32, 4, 11, 0), (32, 4, 14, 0), (32, 4, 8, 128),
                    (50, 1, 36, 245), (50, 2, 21, 256), (50, 3, 16, 613),
                    (64, 2, 32, 256), (64, 3, 21, 16), (64, 3, 21, 256),
                    (64, 3, 21, 403), (64, 4, 52, 2), (75, 1, 60, 16),
                    (75, 2, 32, 4)]))
