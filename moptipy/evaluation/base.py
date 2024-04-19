"""Some internal helper functions and base classes."""

from dataclasses import dataclass
from typing import Any, Callable, Final

from pycommons.types import check_int_range, type_error
from pycommons.version import __version__ as pycommons_version

from moptipy.utils.nputils import rand_seed_check
from moptipy.utils.strings import sanitize_name
from moptipy.version import __version__ as moptipy_version

#: The key for the total number of runs.
KEY_N: Final[str] = "n"
#: a key for the objective function name
KEY_OBJECTIVE_FUNCTION: Final[str] = "objective"
#: a key for the encoding name
KEY_ENCODING: Final[str] = "encoding"

#: The unit of the time axis if time is measured in milliseconds.
TIME_UNIT_MILLIS: Final[str] = "ms"
#: The unit of the time axis of time is measured in FEs
TIME_UNIT_FES: Final[str] = "FEs"

#: The name of the raw objective values data.
F_NAME_RAW: Final[str] = "plainF"
#: The name of the scaled objective values data.
F_NAME_SCALED: Final[str] = "scaledF"
#: The name of the normalized objective values data.
F_NAME_NORMALIZED: Final[str] = "normalizedF"


def check_time_unit(time_unit: Any) -> str:
    """
    Check that the time unit is OK.

    :param time_unit: the time unit
    :return: the time unit string

    >>> check_time_unit("FEs")
    'FEs'
    >>> check_time_unit("ms")
    'ms'
    >>> try:
    ...     check_time_unit(1)
    ... except TypeError as te:
    ...     print(te)
    time_unit should be an instance of str but is int, namely '1'.
    >>> try:
    ...     check_time_unit("blabedibla")
    ... except ValueError as ve:
    ...     print(ve)
    Invalid time unit 'blabedibla', only 'FEs' and 'ms' are permitted.
    """
    if not isinstance(time_unit, str):
        raise type_error(time_unit, "time_unit", str)
    if time_unit in (TIME_UNIT_FES, TIME_UNIT_MILLIS):
        return time_unit
    raise ValueError(
        f"Invalid time unit {time_unit!r}, only {TIME_UNIT_FES!r} "
        f"and {TIME_UNIT_MILLIS!r} are permitted.")


def check_f_name(f_name: Any) -> str:
    """
    Check whether an objective value name is valid.

    :param f_name: the name of the objective function dimension
    :return: the name of the objective function dimension

    >>> check_f_name("plainF")
    'plainF'
    >>> check_f_name("scaledF")
    'scaledF'
    >>> check_f_name("normalizedF")
    'normalizedF'
    >>> try:
    ...     check_f_name(1.0)
    ... except TypeError as te:
    ...     print(te)
    f_name should be an instance of str but is float, namely '1.0'.
    >>> try:
    ...     check_f_name("oops")
    ... except ValueError as ve:
    ...     print(ve)
    Invalid f name 'oops', only 'plainF', 'scaledF', and 'normalizedF' \
are permitted.
    """
    if not isinstance(f_name, str):
        raise type_error(f_name, "f_name", str)
    if f_name in (F_NAME_RAW, F_NAME_SCALED, F_NAME_NORMALIZED):
        return f_name
    raise ValueError(
        f"Invalid f name {f_name!r}, only {F_NAME_RAW!r}, "
        f"{F_NAME_SCALED!r}, and {F_NAME_NORMALIZED!r} are permitted.")


def _set_name(dest: object, name: str, what: str,
              none_allowed: bool = False,
              empty_to_none: bool = True) -> None:
    """
    Check and set a name.

    :param dest: the destination
    :param name: the name to set
    :param what: the name's type
    :param none_allowed: is `None` allowed?
    :param empty_to_none: If both `none_allowed` and `empty_to_none` are
        `True`, then empty strings are converted to `None`

    >>> class TV:
    ...     algorithm: str
    ...     instance: str | None
    >>> t = TV()
    >>> _set_name(t, "bla", "algorithm", False)
    >>> t.algorithm
    'bla'
    >>> _set_name(t, "xbla", "instance", True)
    >>> t.instance
    'xbla'
    >>> _set_name(t, None, "instance", True)
    >>> print(t.instance)
    None
    >>> t.instance = "x"
    >>> _set_name(t, "  ", "instance", True)
    >>> print(t.instance)
    None
    >>> try:
    ...     _set_name(t, 1, "algorithm")
    ... except TypeError as te:
    ...     print(te)
    algorithm name should be an instance of str but is int, namely '1'.
    >>> t.algorithm
    'bla'
    >>> try:
    ...     _set_name(t, "  ", "algorithm")
    ... except ValueError as ve:
    ...     print(ve)
    algorithm name cannot be empty of just consist of white space, but \
'  ' does.
    >>> t.algorithm
    'bla'
    >>> try:
    ...     _set_name(t, "a a", "instance")
    ... except ValueError as ve:
    ...     print(ve)
    Invalid instance name 'a a'.
    >>> print(t.instance)
    None
    >>> try:
    ...     _set_name(t, " ", "instance", True, False)
    ... except ValueError as ve:
    ...     print(ve)
    instance name cannot be empty of just consist of white space, but \
' ' does.
    >>> print(t.instance)
    None
    """
    use_name = name
    if isinstance(name, str):
        use_name = use_name.strip()
        if len(use_name) <= 0:
            if empty_to_none and none_allowed:
                use_name = None
            else:
                raise ValueError(f"{what} name cannot be empty of just cons"
                                 f"ist of white space, but {name!r} does.")
        elif use_name != sanitize_name(use_name):
            raise ValueError(f"Invalid {what} name {name!r}.")
    elif not ((name is None) and none_allowed):
        raise type_error(name, f"{what} name",
                         (str, None) if none_allowed else str)
    object.__setattr__(dest, what, use_name)


class EvaluationDataElement:
    """A base class for all the data classes in this module."""

    def _tuple(self) -> tuple[Any, ...]:
        """
        Create a tuple with all the data of this data class for comparison.

        All the relevant data of an instance of this class is stored in a
        tuple. The tuple is then used in the dunder methods for comparisons.
        The returned tuple *must* be based on the scheme
        `tuple[str, str, str, str, str, int, int, str, str]`.
        They can be shorter than this and they can be longer, but they must
        adhere to this basic scheme:

        1. class name
        2. algorithm name, or `""` if algorithm name is `None`
        3. instance name, or `""` if instance name is `None`
        4. objective name, `""` objective name is `None`
        5. encoding name, or `""` encoding name is `None`
        6. number of runs, or `0` if no number of runs is specified or `1` if
           the data concerns exactly one run
        7. the random seed, or `-1` if no random seed is specified
        8. the string time unit, or `""` if no time unit is given
        9. the scaling name of the objective function, or `""` if no scaling
           name is given

        If the tuples are longer, then all values following after this must be
        integers or floats.

        >>> EvaluationDataElement()._tuple()
        ('EvaluationDataElement',)

        :returns: a tuple with all the data of this class, where `None` values
            are masked out
        """
        return (self.__class__.__name__, )

    def __eq__(self, other) -> bool:
        """
        Compare for `==` with another object based on the `_tuple()` value.

        :param other: the other object to compare to, must be an instance of
            :class:`EvaluationDataElement`
        :retval `True`: if the `other` object's `_tuple()` representation is
            `==` with this object's `_tuple()` representation
        :retval `False`: otherwise
        :raises NotImplementedError: if the other object is not an instance of
            :class:`EvaluationDataElement` and therefore cannot be compared.

        >>> PerRunData("a", "i", "f", "e", 234) == PerRunData(
        ...     "a", "i", "f", "e", 234)
        True
        >>> PerRunData("a", "i", "f", "e", 234) == PerRunData(
        ...     "a", "j", "f", "e", 234)
        False
        >>> try:
        ...     PerRunData("a", "i", "f", "e", 234) == 3
        ... except NotImplementedError as ni:
        ...     print(ni)
        Cannot compare PerRunData(algorithm='a', instance='i', \
objective='f', encoding='e', rand_seed=234) with 3 for ==.
        """
        if isinstance(other, EvaluationDataElement):
            return self._tuple() == other._tuple()
        raise NotImplementedError(
            f"Cannot compare {self} with {other} for ==.")

    def __ne__(self, other) -> bool:
        """
        Compare for `!=` with another object based on the `_tuple()` value.

        :param other: the other object to compare to, must be an instance of
            :class:`EvaluationDataElement`
        :retval `True`: if the `other` object's `_tuple()` representation is
            `!=` with this object's `_tuple()` representation
        :retval `False`: otherwise
        :raises NotImplementedError: if the other object is not an instance of
            :class:`EvaluationDataElement` and therefore cannot be compared.

        >>> PerRunData("a", "i", "f", "e", 234) != PerRunData(
        ...     "a", "i", "f", "e", 234)
        False
        >>> PerRunData("a", "i", "f", "e", 234) != PerRunData(
        ...     "a", "j", "f", "e", 234)
        True
        >>> try:
        ...     PerRunData("a", "i", "f", "e", 234) != 3
        ... except NotImplementedError as ni:
        ...     print(ni)
        Cannot compare PerRunData(algorithm='a', instance='i', \
objective='f', encoding='e', rand_seed=234) with 3 for !=.
        """
        if isinstance(other, EvaluationDataElement):
            return self._tuple() != other._tuple()
        raise NotImplementedError(
            f"Cannot compare {self} with {other} for !=.")

    def __lt__(self, other) -> bool:
        """
        Compare for `<` with another object based on the `_tuple()` value.

        :param other: the other object to compare to, must be an instance of
            :class:`EvaluationDataElement`
        :retval `True`: if the `other` object's `_tuple()` representation is
            `<` with this object's `_tuple()` representation
        :retval `False`: otherwise
        :raises NotImplementedError: if the other object is not an instance of
            :class:`EvaluationDataElement` and therefore cannot be compared.

        >>> PerRunData("a", "i", "f", "e", 234) < PerRunData(
        ...     "a", "i", "f", "e", 234)
        False
        >>> PerRunData("a", "i", "f", "e", 234) < PerRunData(
        ...     "a", "j", "f", "e", 234)
        True
        >>> PerRunData("a", "j", "f", "e", 234) < PerRunData(
        ...     "a", "i", "f", "e", 234)
        False
        >>> try:
        ...     PerRunData("a", "i", "f", "e", 234) < 3
        ... except NotImplementedError as ni:
        ...     print(ni)
        Cannot compare PerRunData(algorithm='a', instance='i', \
objective='f', encoding='e', rand_seed=234) with 3 for <.
        """
        if isinstance(other, EvaluationDataElement):
            return self._tuple() < other._tuple()
        raise NotImplementedError(
            f"Cannot compare {self} with {other} for <.")

    def __le__(self, other) -> bool:
        """
        Compare for `<=` with another object based on the `_tuple()` value.

        :param other: the other object to compare to, must be an instance of
            :class:`EvaluationDataElement`
        :retval `True`: if the `other` object's `_tuple()` representation is
            `<=` with this object's `_tuple()` representation
        :retval `False`: otherwise
        :raises NotImplementedError: if the other object is not an instance of
            :class:`EvaluationDataElement` and therefore cannot be compared.

        >>> PerRunData("a", "i", "f", "e", 234) <= PerRunData(
        ...     "a", "i", "f", "e", 234)
        True
        >>> PerRunData("a", "i", "f", "e", 234) <= PerRunData(
        ...     "a", "j", "f", "e", 234)
        True
        >>> PerRunData("a", "j", "f", "e", 234) < PerRunData(
        ...     "a", "i", "f", "e", 234)
        False
        >>> try:
        ...     PerRunData("a", "i", "f", "e", 234) <= 3
        ... except NotImplementedError as ni:
        ...     print(ni)
        Cannot compare PerRunData(algorithm='a', instance='i', \
objective='f', encoding='e', rand_seed=234) with 3 for <=.
        """
        if isinstance(other, EvaluationDataElement):
            return self._tuple() <= other._tuple()
        raise NotImplementedError(
            f"Cannot compare {self} with {other} for <=.")

    def __gt__(self, other) -> bool:
        """
        Compare for `>` with another object based on the `_tuple()` value.

        :param other: the other object to compare to, must be an instance of
            :class:`EvaluationDataElement`
        :retval `True`: if the `other` object's `_tuple()` representation is
            `>` with this object's `_tuple()` representation
        :retval `False`: otherwise
        :raises NotImplementedError: if the other object is not an instance of
            :class:`EvaluationDataElement` and therefore cannot be compared.

        >>> PerRunData("a", "i", "f", "e", 234) > PerRunData(
        ...     "a", "i", "f", "e", 234)
        False
        >>> PerRunData("a", "i", "f", "e", 234) > PerRunData(
        ...     "a", "j", "f", "e", 234)
        False
        >>> PerRunData("a", "j", "f", "e", 234) > PerRunData(
        ...     "a", "i", "f", "e", 234)
        True
        >>> try:
        ...     PerRunData("a", "i", "f", "e", 234) > 3
        ... except NotImplementedError as ni:
        ...     print(ni)
        Cannot compare PerRunData(algorithm='a', instance='i', \
objective='f', encoding='e', rand_seed=234) with 3 for >.
        """
        if isinstance(other, EvaluationDataElement):
            return self._tuple() > other._tuple()
        raise NotImplementedError(
            f"Cannot compare {self} with {other} for >.")

    def __ge__(self, other) -> bool:
        """
        Compare for `>=` with another object based on the `_tuple()` value.

        :param other: the other object to compare to, must be an instance of
            :class:`EvaluationDataElement`
        :retval `True`: if the `other` object's `_tuple()` representation is
            `>=` with this object's `_tuple()` representation
        :retval `False`: otherwise
        :raises NotImplementedError: if the other object is not an instance of
            :class:`EvaluationDataElement` and therefore cannot be compared.

        >>> PerRunData("a", "i", "f", "e", 234) >= PerRunData(
        ...     "a", "i", "f", "e", 234)
        True
        >>> PerRunData("a", "i", "f", "e", 234) >= PerRunData(
        ...     "a", "j", "f", "e", 234)
        False
        >>> PerRunData("a", "j", "f", "e", 234) >= PerRunData(
        ...     "a", "i", "f", "e", 234)
        True
        >>> try:
        ...     PerRunData("a", "i", "f", "e", 234) >= 3
        ... except NotImplementedError as ni:
        ...     print(ni)
        Cannot compare PerRunData(algorithm='a', instance='i', \
objective='f', encoding='e', rand_seed=234) with 3 for >=.
        """
        if isinstance(other, EvaluationDataElement):
            return self._tuple() >= other._tuple()
        raise NotImplementedError(
            f"Cannot compare {self} with {other} for >=.")


@dataclass(frozen=True, init=False, order=False, eq=False)
class PerRunData(EvaluationDataElement):
    """
    An immutable record of information over a single run.

    >>> p = PerRunData("a", "i", "f", None, 234)
    >>> p.instance
    'i'
    >>> p.algorithm
    'a'
    >>> p.objective
    'f'
    >>> print(p.encoding)
    None
    >>> p.rand_seed
    234
    >>> p = PerRunData("a", "i", "f", "e", 234)
    >>> p.instance
    'i'
    >>> p.algorithm
    'a'
    >>> p.objective
    'f'
    >>> p.encoding
    'e'
    >>> p.rand_seed
    234
    >>> try:
    ...     PerRunData(3, "i", "f", "e", 234)
    ... except TypeError as te:
    ...     print(te)
    algorithm name should be an instance of str but is int, namely '3'.
    >>> try:
    ...     PerRunData("@1 2", "i", "f", "e", 234)
    ... except ValueError as ve:
    ...     print(ve)
    Invalid algorithm name '@1 2'.
    >>> try:
    ...     PerRunData("x", 3.2, "f", "e", 234)
    ... except TypeError as te:
    ...     print(te)
    instance name should be an instance of str but is float, namely '3.2'.
    >>> try:
    ...     PerRunData("x", "sdf i", "f", "e", 234)
    ... except ValueError as ve:
    ...     print(ve)
    Invalid instance name 'sdf i'.
    >>> try:
    ...     PerRunData("a", "i", True, "e", 234)
    ... except TypeError as te:
    ...     print(te)
    objective name should be an instance of str but is bool, namely 'True'.
    >>> try:
    ...     PerRunData("x", "i", "d-f", "e", 234)
    ... except ValueError as ve:
    ...     print(ve)
    Invalid objective name 'd-f'.
    >>> try:
    ...     PerRunData("x", "i", "f", 54.2, 234)
    ... except TypeError as te:
    ...     print(te)
    encoding name should be an instance of any in {None, str} but is float, \
namely '54.2'.
    >>> try:
    ...     PerRunData("y", "i", "f", "x  x", 234)
    ... except ValueError as ve:
    ...     print(ve)
    Invalid encoding name 'x  x'.
    >>> try:
    ...     PerRunData("x", "i", "f", "e", 3.3)
    ... except TypeError as te:
    ...     print(te)
    rand_seed should be an instance of int but is float, namely '3.3'.
    >>> try:
    ...     PerRunData("x", "i", "f", "e", -234)
    ... except ValueError as ve:
    ...     print(ve)
    rand_seed=-234 is invalid, must be in 0..18446744073709551615.
    """

    #: The algorithm that was applied.
    algorithm: str
    #: The problem instance that was solved.
    instance: str
    #: the name of the objective function
    objective: str
    #: the encoding, if any, or `None` if no encoding was used
    encoding: str | None
    #: The seed of the random number generator.
    rand_seed: int

    def __init__(self, algorithm: str, instance: str, objective: str,
                 encoding: str | None, rand_seed: int):
        """
        Create a per-run data record.

        :param algorithm: the algorithm name
        :param instance: the instance name
        :param objective: the name of the objective function
        :param encoding: the name of the encoding that was used, if any, or
            `None` if no encoding was used
        :param rand_seed: the random seed
        """
        _set_name(self, algorithm, "algorithm")
        _set_name(self, instance, "instance")
        _set_name(self, objective, "objective")
        _set_name(self, encoding, "encoding", True, False)
        object.__setattr__(self, "rand_seed", rand_seed_check(rand_seed))

    def _tuple(self) -> tuple[Any, ...]:
        """
        Get the tuple representation of this object used in comparisons.

        :return: the comparison-relevant data of this object in a tuple

        >>> PerRunData("a", "i", "f", "e", 234)._tuple()
        ('PerRunData', 'a', 'i', 'f', 'e', 1, 234)
        >>> PerRunData("a", "i", "f", None, 234)._tuple()
        ('PerRunData', 'a', 'i', 'f', '', 1, 234)
        """
        return (self.__class__.__name__, self.algorithm, self.instance,
                self.objective,
                "" if self.encoding is None else self.encoding, 1,
                self.rand_seed)


@dataclass(frozen=True, init=False, order=False, eq=False)
class MultiRunData(EvaluationDataElement):
    """
    A class that represents statistics over a set of runs.

    If one algorithm*instance is used, then `algorithm` and `instance` are
    defined. Otherwise, only the parameter which is the same over all recorded
    runs is defined.

    >>> p = MultiRunData("a", "i", "f", None, 3)
    >>> p.instance
    'i'
    >>> p.algorithm
    'a'
    >>> p.objective
    'f'
    >>> print(p.encoding)
    None
    >>> p.n
    3
    >>> p = MultiRunData(None, None, None, "x", 3)
    >>> print(p.instance)
    None
    >>> print(p.algorithm)
    None
    >>> print(p.objective)
    None
    >>> p.encoding
    'x'
    >>> p.n
    3
    >>> try:
    ...     MultiRunData(1, "i", "f", "e", 234)
    ... except TypeError as te:
    ...     print(te)
    algorithm name should be an instance of any in {None, str} but is int, \
namely '1'.
    >>> try:
    ...     MultiRunData("x x", "i", "f", "e", 234)
    ... except ValueError as ve:
    ...     print(ve)
    Invalid algorithm name 'x x'.
    >>> try:
    ...     MultiRunData("a", 5.5, "f", "e", 234)
    ... except TypeError as te:
    ...     print(te)
    instance name should be an instance of any in {None, str} but is float, \
namely '5.5'.
    >>> try:
    ...     MultiRunData("x", "a-i", "f", "e", 234)
    ... except ValueError as ve:
    ...     print(ve)
    Invalid instance name 'a-i'.
    >>> try:
    ...     MultiRunData("a", "i", True, "e", 234)
    ... except TypeError as te:
    ...     print(te)
    objective name should be an instance of any in {None, str} but is bool, \
namely 'True'.
    >>> try:
    ...     MultiRunData("xx", "i", "d'@f", "e", 234)
    ... except ValueError as ve:
    ...     print(ve)
    Invalid objective name "d'@f".
    >>> try:
    ...     MultiRunData("yy", "i", "f", -9.4, 234)
    ... except TypeError as te:
    ...     print(te)
    encoding name should be an instance of any in {None, str} but is float, \
namely '-9.4'.
    >>> try:
    ...     MultiRunData("xx", "i", "f", "e-{a", 234)
    ... except ValueError as ve:
    ...     print(ve)
    Invalid encoding name 'e-{a'.
    >>> try:
    ...     MultiRunData("x", "i", "f", "e", -1.234)
    ... except TypeError as te:
    ...     print(te)
    n should be an instance of int but is float, namely '-1.234'.
    >>> try:
    ...     MultiRunData("xx", "i", "f", "e", 1_000_000_000_000_000_000_000)
    ... except ValueError as ve:
    ...     print(ve)
    n=1000000000000000000000 is invalid, must be in 1..1000000000000000.
    """

    #: The algorithm that was applied, if the same over all runs.
    algorithm: str | None
    #: The problem instance that was solved, if the same over all runs.
    instance: str | None
    #: the name of the objective function, if the same over all runs
    objective: str | None
    #: the encoding, if any, or `None` if no encoding was used or if it was
    #: not the same over all runs
    encoding: str | None
    #: The number of runs over which the statistic information is computed.
    n: int

    def __init__(self, algorithm: str | None, instance: str | None,
                 objective: str | None, encoding: str | None, n: int):
        """
        Create the dataset of an experiment-setup combination.

        :param algorithm: the algorithm name, if all runs are with the same
            algorithm, `None` otherwise
        :param instance: the instance name, if all runs are on the same
            instance, `None` otherwise
        :param objective: the objective name, if all runs are on the same
            objective function, `None` otherwise
        :param encoding: the encoding name, if all runs are on the same
            encoding and an encoding was actually used, `None` otherwise
        :param n: the total number of runs
        """
        _set_name(self, algorithm, "algorithm", True, False)
        _set_name(self, instance, "instance", True, False)
        _set_name(self, objective, "objective", True, False)
        _set_name(self, encoding, "encoding", True, False)
        object.__setattr__(self, "n", check_int_range(
            n, "n", 1, 1_000_000_000_000_000))

    def _tuple(self) -> tuple[Any, ...]:
        """
        Get the tuple representation of this object used in comparisons.

        :return: the comparison-relevant data of this object in a tuple

        >>> MultiRunData("a", "i", "f", None, 3)._tuple()
        ('MultiRunData', 'a', 'i', 'f', '', 3, -1)
        >>> MultiRunData(None, "i", "f", "e", 31)._tuple()
        ('MultiRunData', '', 'i', 'f', 'e', 31, -1)
        >>> MultiRunData("x", None, "fy", "e1", 131)._tuple()
        ('MultiRunData', 'x', '', 'fy', 'e1', 131, -1)
        >>> MultiRunData("yx", "z", None, "xe1", 2131)._tuple()
        ('MultiRunData', 'yx', 'z', '', 'xe1', 2131, -1)
        """
        return (self.__class__.__name__,
                "" if self.algorithm is None else self.algorithm,
                "" if self.instance is None else self.instance,
                "" if self.objective is None else self.objective,
                "" if self.encoding is None else self.encoding,
                self.n, -1)


@dataclass(frozen=True, init=False, order=False, eq=False)
class MultiRun2DData(MultiRunData):
    """
    A multi-run data based on one time and one objective dimension.

    >>> p = MultiRun2DData("a", "i", "f", None, 3,
    ...                    TIME_UNIT_FES, F_NAME_SCALED)
    >>> p.instance
    'i'
    >>> p.algorithm
    'a'
    >>> p.objective
    'f'
    >>> print(p.encoding)
    None
    >>> p.n
    3
    >>> print(p.time_unit)
    FEs
    >>> print(p.f_name)
    scaledF
    >>> try:
    ...     MultiRun2DData("a", "i", "f", None, 3,
    ...                    3, F_NAME_SCALED)
    ... except TypeError as te:
    ...     print(te)
    time_unit should be an instance of str but is int, namely '3'.
    >>> try:
    ...     MultiRun2DData("a", "i", "f", None, 3,
    ...                    "sdfjsdf", F_NAME_SCALED)
    ... except ValueError as ve:
    ...     print(ve)
    Invalid time unit 'sdfjsdf', only 'FEs' and 'ms' are permitted.
    >>> try:
    ...     MultiRun2DData("a", "i", "f", None, 3,
    ...                    TIME_UNIT_FES, True)
    ... except TypeError as te:
    ...     print(te)
    f_name should be an instance of str but is bool, namely 'True'.
    >>> try:
    ...     MultiRun2DData("a", "i", "f", None, 3,
    ...                    TIME_UNIT_FES, "blablue")
    ... except ValueError as ve:
    ...     print(ve)
    Invalid f name 'blablue', only 'plainF', 'scaledF', and 'normalizedF' \
are permitted.
    """

    #: The unit of the time axis.
    time_unit: str
    #: the name of the objective value axis.
    f_name: str

    def __init__(self, algorithm: str | None, instance: str | None,
                 objective: str | None, encoding: str | None, n: int,
                 time_unit: str, f_name: str):
        """
        Create multi-run data based on one time and one objective dimension.

        :param algorithm: the algorithm name, if all runs are with the same
            algorithm
        :param instance: the instance name, if all runs are on the same
            instance
        :param objective: the objective name, if all runs are on the same
            objective function, `None` otherwise
        :param encoding: the encoding name, if all runs are on the same
            encoding and an encoding was actually used, `None` otherwise
        :param n: the total number of runs
        :param time_unit: the time unit
        :param f_name: the objective dimension name
        """
        super().__init__(algorithm, instance, objective, encoding, n)
        object.__setattr__(self, "time_unit", check_time_unit(time_unit))
        object.__setattr__(self, "f_name", check_f_name(f_name))

    def _tuple(self) -> tuple[Any, ...]:
        """
        Get the tuple representation of this object used in comparisons.

        :return: the comparison-relevant data of this object in a tuple

        >>> MultiRun2DData("a", "i", "f", None, 3,
        ...                TIME_UNIT_FES, F_NAME_SCALED)._tuple()
        ('MultiRun2DData', 'a', 'i', 'f', '', 3, -1, 'FEs', 'scaledF')
        >>> MultiRun2DData(None, "ix", None, "x", 43,
        ...                TIME_UNIT_MILLIS, F_NAME_RAW)._tuple()
        ('MultiRun2DData', '', 'ix', '', 'x', 43, -1, 'ms', 'plainF')
        >>> MultiRun2DData("xa", None, None, None, 143,
        ...                TIME_UNIT_MILLIS, F_NAME_NORMALIZED)._tuple()
        ('MultiRun2DData', 'xa', '', '', '', 143, -1, 'ms', 'normalizedF')
        """
        return (self.__class__.__name__,
                "" if self.algorithm is None else self.algorithm,
                "" if self.instance is None else self.instance,
                "" if self.objective is None else self.objective,
                "" if self.encoding is None else self.encoding,
                self.n, -1, self.time_unit, self.f_name)


def get_instance(obj: PerRunData | MultiRunData) -> str | None:
    """
    Get the instance of a given object.

    :param obj: the object
    :return: the instance string, or `None` if no instance is specified

    >>> p1 = MultiRunData("a", "i1", None, "x", 3)
    >>> get_instance(p1)
    'i1'
    >>> p2 = PerRunData("a", "i2", "f", "x", 31)
    >>> get_instance(p2)
    'i2'
    """
    return obj.instance


def get_algorithm(obj: PerRunData | MultiRunData) -> str | None:
    """
    Get the algorithm of a given object.

    :param obj: the object
    :return: the algorithm string, or `None` if no algorithm is specified

    >>> p1 = MultiRunData("a1", "i1", "f", "y", 3)
    >>> get_algorithm(p1)
    'a1'
    >>> p2 = PerRunData("a2", "i2", "y", None, 31)
    >>> get_algorithm(p2)
    'a2'
    """
    return obj.algorithm


def sort_key(obj: PerRunData | MultiRunData) -> tuple[Any, ...]:
    """
    Get the default sort key for the given object.

    The sort key is a tuple with well-defined field elements that should
    allow for a default and consistent sorting over many different elements of
    the experiment evaluation data API. Sorting should work also for lists
    containing elements of different classes.

    :param obj: the object
    :return: the sort key

    >>> p1 = MultiRunData("a1", "i1", "f", None, 3)
    >>> p2 = PerRunData("a2", "i2", "f", None, 31)
    >>> sort_key(p1) < sort_key(p2)
    True
    >>> sort_key(p1) >= sort_key(p2)
    False
    >>> p3 = MultiRun2DData("a", "i", "f", None, 3,
    ...                     TIME_UNIT_FES, F_NAME_SCALED)
    >>> sort_key(p3) < sort_key(p1)
    True
    >>> sort_key(p3) >= sort_key(p1)
    False
    """
    # noinspection PyProtectedMember
    return obj._tuple()


def _csv_motipy_footer(dest: Callable[[str], Any]) -> None:
    """
    Print the standard csv footer.

    :param dest: the destination to write to
    """
    dest("")
    dest("This data has been generated with moptipy version "
         f"{moptipy_version} using pycommons version "
         f"{pycommons_version}.")
    dest("You can find moptipy at https://thomasweise.github.io/mopitpy.")
    dest(
        "You can find pycommons at https://thomasweise.github.io/pycommons.")


#: a description of the algorithm field
DESC_ALGORITHM: Final[str] = "the name of the algorithm setup that was used."
#: a description of the instance field
DESC_INSTANCE: Final[str] = ("the name of the problem instance to which the "
                             "algorithm was applied.")
#: a description of the objective function field
DESC_OBJECTIVE_FUNCTION: Final[str] = \
    ("the name of the objective function (often also called fitness function "
     "or cost function) that was used to rate the solution quality.")
#: a description of the encoding field
DESC_ENCODING: Final[str] = \
    ("the name of the encoding, often also called genotype-phenotype mapping"
     ", used. In some problems, the search space on which the algorithm "
     "works is different from the space of possible solutions. For example, "
     "when solving a scheduling problem, maybe our optimization algorithm "
     "navigates in the space of permutations, but the solutions are Gantt "
     "charts. The encoding is the function that translates the points in "
     "the search space (e.g., permutations) to the points in the solution "
     "space (e.g., Gantt charts). Nothing if no encoding was used.")
