"""Routines for handling strings."""

from math import isnan, isfinite, inf
from re import sub
from typing import Union, Optional, Final, Iterable, List, cast, Callable

from moptipy.utils.math import __try_int, try_int
from moptipy.utils.types import type_error


def float_to_str(x: float) -> str:
    """
    Convert `float` to a string.

    :param x: the floating point value
    :return: the string representation

    >>> float_to_str(1.3)
    '1.3'
    >>> float_to_str(1.0)
    '1'
    """
    if x == 0:
        return "0"
    s = repr(x)
    if isnan(x):
        raise ValueError(f"'{s}' not permitted.")
    if s.endswith(".0"):
        return s[:-2]
    return s


def bool_to_str(value: bool) -> str:
    """
    Convert a Boolean value to a string.

    :param value: the Boolean value
    :return: the string

    >>> print(bool_to_str(True))
    T
    >>> print(bool_to_str(False))
    F
    """
    return 'T' if value else 'F'


def str_to_bool(value: str) -> bool:
    """
    Convert a string to a boolean value.

    :param value: the string value
    :return: the boolean value

    >>> str_to_bool("T")
    True
    >>> str_to_bool("F")
    False
    >>> try:
    ...     str_to_bool("x")
    ... except ValueError as v:
    ...     print(v)
    Expected 'T' or 'F', but got 'x'.
    """
    if value == "T":
        return True
    if value == "F":
        return False
    raise ValueError(f"Expected 'T' or 'F', but got '{value}'.")


def num_to_str(value: Union[int, float]) -> str:
    """
    Transform a numerical type to a string.

    :param value: the value
    :return: the string

    >>> num_to_str(1)
    '1'
    >>> num_to_str(1.5)
    '1.5'
    """
    return str(value) if isinstance(value, int) else float_to_str(value)


def intfloatnone_to_str(val: Union[int, float, None]) -> str:
    """
    Convert an integer ot float or `None` to a string.

    :param val: the value
    :return: the string representation
    :rtype: str

    >>> print(repr(intfloatnone_to_str(None)))
    ''
    >>> print(intfloatnone_to_str(12))
    12
    >>> print(intfloatnone_to_str(12.3))
    12.3
    """
    return "" if val is None else num_to_str(val)


def intnone_to_str(val: Optional[int]) -> str:
    """
    Convert an integer or `None` to a string.

    :param val: the value
    :return: the string representation

    >>> print(repr(intnone_to_str(None)))
    ''
    >>> print(intnone_to_str(12))
    12
    """
    return "" if val is None else str(val)


def str_to_intfloat(val: str) -> Union[int, float]:
    """
    Convert a string to an int or float.

    :param val: the string value
    :return: the int or float

    >>> print(type(str_to_intfloat("15.0")))
    <class 'int'>
    >>> print(type(str_to_intfloat("15.1")))
    <class 'float'>
    """
    return __try_int(float(val)) if ("e" in val) or \
                                    ("E" in val) or \
                                    ("." in val) or \
                                    ("inf" in val) else int(val)


def str_to_intfloatnone(val: str) -> Union[int, float, None]:
    """
    Convert a string to an int or float or None.

    :param val: the string value
    :return: the int or float or None

    >>> print(str_to_intfloatnone(""))
    None
    >>> print(type(str_to_intfloatnone("5.0")))
    <class 'int'>
    >>> print(type(str_to_intfloatnone("5.1")))
    <class 'float'>
    """
    return None if len(val) <= 0 else str_to_intfloat(val)


def str_to_intnone(val: str) -> Optional[int]:
    """
    Convert a string to an int or None.

    :param val: the string value
    :return: the int or None

    >>> print(str_to_intnone(""))
    None
    >>> print(str_to_intnone("5"))
    5
    """
    return None if len(val) <= 0 else int(val)


def replace_all(find: str, replace: str, src: str) -> str:
    """
    Perform a recursive replacement of strings.

    After applying this function, there will not be any occurence of `find`
    left in `src`. All of them will have been replaced by `replace`. If that
    produces new instances of `find`, these will be replaced as well.
    If `replace` contains `find`, this will lead to an endless loop!

    :param find: the string to find
    :param replace: the string with which it will be replaced
    :param src: the string in which we search
    :return: the string `src`, with all occurrences of find replaced by replace

    >>> replace_all("a", "b", "abc")
    'bbc'
    >>> replace_all("aa", "a", "aaaaa")
    'a'
    >>> replace_all("aba", "a", "abaababa")
    'aa'
    """
    new_len = len(src)
    while True:
        src = src.replace(find, replace)
        old_len = new_len
        new_len = len(src)
        if new_len >= old_len:
            return src


def __replace_double(replace: str, src: str) -> str:
    """
    Replace any double-occurrence of a string with a single occurrence.

    :param replace: the string to replace
    :param src: the source string
    :returns: the updated string
    """
    return replace_all(replace + replace, replace, src)


#: the separator of different filename parts
PART_SEPARATOR: Final[str] = "_"
#: the replacement for "." in a file name
DECIMAL_DOT_REPLACEMENT: Final[str] = "d"


def sanitize_name(name: str) -> str:
    """
    Sanitize a name in such a way that it can be used as path component.

    >>> sanitize_name(" hello world ")
    'hello_world'
    >>> sanitize_name(" 56.6-455 ")
    '56d6-455'
    >>> sanitize_name(" _ i _ am _ funny   --6 _ ")
    'i_am_funny_-6'

    :param name: the name that should be sanitized
    :return: the sanitized name
    :raises ValueError: if the name is invalid or empty
    :raises TypeError: if the name is `None` or not a string
    """
    if not isinstance(name, str):
        raise type_error(name, "name", str)
    orig_name = name
    name = name.strip()
    name = __replace_double("-", name)
    name = __replace_double("_", name)
    name = __replace_double(".", name).replace(".", DECIMAL_DOT_REPLACEMENT)

    name = sub(r"[^\w\s-]", '', name)
    name = sub(r"\s+", PART_SEPARATOR, name)
    name = __replace_double("_", name)

    if name.startswith("_"):
        name = name[1:]

    if name.endswith("_"):
        name = name[:len(name) - 1]

    if len(name) <= 0:
        raise ValueError(
            f"Sanitized name must not become empty, but '{orig_name}' does.")

    return name


def sanitize_names(names: Iterable[str]) -> str:
    """
    Sanitize a set of names.

    >>> sanitize_names(["", " sdf ", "", "5-3"])
    'sdf_5-3'
    >>> sanitize_names([" a ", " b", " c", "", "6", ""])
    'a_b_c_6'

    :param names: the list of names.
    :return: the sanitized name
    """
    return PART_SEPARATOR.join([
        sanitize_name(name) for name in names if len(name) > 0])


#: the internal integer to float conversion threshold
__INT_TO_FLOAT_THRESHOLD: Final[float] = 1E10


def default_float_format(min_finite: Union[int, float] = 0,
                         max_finite: Union[int, float] = 0,
                         frac_len: int = 2) -> str:
    """
    Get the default float format.

    :param min_finite: the minimum finite value
    :param max_finite: the maximum finite value
    :param frac_len: the longest fraction
    """
    if not isinstance(min_finite, (int, float)):
        raise type_error(min_finite, "min_finite", (int, float))
    if not isinstance(max_finite, (int, float)):
        raise type_error(max_finite, "max_finite", (int, float))
    if not (isfinite(min_finite) and isfinite(max_finite)
            and (min_finite <= max_finite)):
        raise ValueError("invalid min_finite, max_finite pair "
                         f"{min_finite}, {max_finite}.")
    if not isinstance(frac_len, int):
        raise type_error(frac_len, "frac_len", int)
    if not (0 <= frac_len < 100):
        raise ValueError(f"invalid frac_len {frac_len}.")

    if ((-__INT_TO_FLOAT_THRESHOLD) <= min_finite) \
            and (max_finite <= __INT_TO_FLOAT_THRESHOLD):
        if (frac_len <= 0) or (min_finite <= -1E4) or (max_finite >= 1E4):
            return "{:.0f}"
        if (frac_len <= 1) or (min_finite <= -1E3) or (max_finite >= 1E3):
            return "{:.1f}"
        if (frac_len <= 2) or (min_finite <= -1E2) or (max_finite >= 1E2):
            return "{:.2f}"
        return "{:.3f}"
    return "{:.2e}"


def numbers_to_strings(source: Union[int, float, None,
                                     Iterable[Union[int, float, None]]],
                       none_str: Optional[str] = None,
                       nan_str: Optional[str] = r"$\emptyset$",
                       positive_infty_str: Optional[str] = r"$\infty$",
                       negative_infty_str: Optional[str] = r"$-\infty$",
                       int_renderer: Optional[Callable[[int], str]] = None,
                       float_format_getter: Callable[
                           [Union[int, float], Union[int, float], int], str]
                       = default_float_format,
                       exponent_renderer: Callable[[str], str] =
                       lambda e: f"*10^{e}^",
                       int_to_float_threshold:
                       Union[int, float] = __INT_TO_FLOAT_THRESHOLD) \
        -> List[Optional[str]]:
    r"""
    Convert a numerical column to text with uniform shape.

    Often, we need to convert a set of numbers to strings as output for a
    table or another representative thext. In such a case, you want to present
    all numbers in the set in the same format.
    Imagine you have the number vector `[1E-4, 1/7, 123456789012345678]`. If
    you simply convert this list to a string directly, what you get is
    `[0.0001, 0.14285714285714285, 123456789012345678]`. Now this looks very
    ugly. First, we have one very big number `123456789012345678`. If the
    numbers stem from an experiment, then we hardly are able to obtain any
    number at a very extreme precision. The 18 digits in `123456789012345678`
    sort of suggest a precision to 18 decimals, since the number ends in
    specific digits (as opposed to `123450000000000000` which a reader would
    naturally preceive as a rounded quantity). Additionally, we the number
    `0.14285714285714285`, which has a very long fractional part, which, too,
    suggests a very high precision. Writing both mentioned numbers next to
    each other, this suggests as if we could present a number as high as
    10**18 at a precision of 10**-17. And it also looks ugly, because both
    numbers are not uniformly formatted. Instead, our function here renders
    the number list as `['1.00*10^-4^', '1.43*10^-1^', '1.23*10^17^']`. It
    recognizes that we should present numbers as powers of ten and then limits
    the precision to three digits.

    This function is thus intended to produce some sort of uniform format with
    reasonable precision uniformly for a numerical vector, under the
    assumption that all numbers should be presented in the same numerical range
    and quantity.

    :param source: the column data
    :param none_str: the string replacement for `None`
    :param nan_str: the string to be used for NaN
    :param positive_infty_str: the string to be used for positive infinity
    :param negative_infty_str: the string to be used for negative infinity
    :param int_renderer: the renderer for integers
    :param float_format_getter: a float format getter
    :param exponent_renderer: the renderer for exponents
    :param int_to_float_threshold: the absolute threshold after which integers
        will be force-converted to floating point numbers to ensure that we do
        not express uselessly large numbers as integers
    :returns: a list with the text representation

    >>> from moptipy.utils.lang import EN
    >>> EN.set_current()
    >>> numbers_to_strings([1.75651, 212, 3234234])
    ['2', '212', "3'234'234"]
    >>> numbers_to_strings([1.75651, 22, 34])
    ['1.757', '22.000', '34.000']
    >>> numbers_to_strings([1.75651, 122, 34])
    ['1.76', '122.00', '34.00']
    >>> numbers_to_strings([1.75651, 122, 3334])
    ['1.8', '122.0', "3'334.0"]
    >>> numbers_to_strings([1.5, 212, 3234234])
    ['2', '212', "3'234'234"]
    >>> numbers_to_strings([1.5, 2e12, 3234234])
    ['1.50*10^0^', '2.00*10^12^', '3.23*10^6^']
    >>> numbers_to_strings([233, 22139283482834, 3234234])
    ['2.33*10^2^', '2.21*10^13^', '3.23*10^6^']
    >>> numbers_to_strings([233, 22139283, 3234234])
    ['233', "22'139'283", "3'234'234"]
    >>> from math import nan, inf
    >>> numbers_to_strings([22139283, inf, -inf, nan, None])
    ["22'139'283", '$\\infty$', '$-\\infty$', '$\\emptyset$', None]
    >>> numbers_to_strings([1E-4, 1/7, 123456789012345678])
    ['1.00*10^-4^', '1.43*10^-1^', '1.23*10^17^']
    """
    # perform type checks
    if (source is None) or isinstance(source, (int, float)):
        source = [source]
    if not isinstance(source, Iterable):
        raise type_error(source, "source", Iterable)
    if (none_str is not None) and (not isinstance(none_str, str)):
        raise type_error(none_str, "none_str", (str, None))
    if (nan_str is not None) and (not isinstance(nan_str, str)):
        raise type_error(nan_str, "nan_str", (str, None))
    if (positive_infty_str is not None) \
            and (not isinstance(positive_infty_str, str)):
        raise type_error(positive_infty_str, "positive_infty_str",
                         (str, None))
    if (negative_infty_str is not None) \
            and (not isinstance(negative_infty_str, str)):
        raise type_error(negative_infty_str, "negative_infty_str",
                         (str, None))
    if int_renderer is None:
        from moptipy.utils.lang import Lang  # pylint: disable=C0415,R0401
        int_renderer = Lang.current().format_int
    if not callable(int_renderer):
        raise type_error(int_renderer, "int_renderer", call=True)
    if not callable(float_format_getter):
        raise type_error(float_format_getter,
                         "float_format_getter", call=True)
    if not callable(exponent_renderer):
        raise type_error(exponent_renderer,
                         "exponent_renderer", call=True)
    if not isinstance(int_to_float_threshold, (int, float)):
        raise type_error(int_to_float_threshold, "int_to_float_threshold",
                         (int, float))
    if isnan(int_to_float_threshold) or (int_to_float_threshold <= 0):
        raise ValueError(
            f"invalid int_to_float_threshold: {int_to_float_threshold}.")

    # step one: get the raw numerical data
    data: Final[List[Union[int, float, None]]] = \
        cast(List, source) if isinstance(source, List) else list(source)
    dlen: Final[int] = len(data)
    if dlen <= 0:
        raise ValueError("Data cannot be empty.")

    # step two: investigate the data ranges and structure
    all_is_none: bool = True
    all_is_int: bool = True
    max_finite: Union[int, float] = -inf
    min_finite: Union[int, float] = inf
    longest_fraction: int = -1

    for i, d in enumerate(data):
        if d is None:
            continue
        all_is_none = False
        d2 = try_int(d) if isfinite(d) else d
        if isinstance(d2, int):
            if d2 < min_finite:
                min_finite = d2
            if d2 > max_finite:
                max_finite = d2
            if not ((-int_to_float_threshold) <= d2
                    <= int_to_float_threshold):
                d2 = float(d2)
        if d2 is not d:
            data[i] = d2

        if isfinite(d2):
            if not isinstance(d2, int):
                all_is_int = False
                s = str(d2)
                if not (("E" in s) or ("e" in s)):
                    i = s.find(".")
                    if i >= 0:
                        i = len(s) - i - 1
                        if i > longest_fraction:
                            longest_fraction = i
            if d2 < min_finite:
                min_finite = d2
            if d2 > max_finite:
                max_finite = d2

    # step three: if all data is None, we can return here
    if all_is_none:
        return [none_str] * dlen

    # create the protected integer renderer
    def __toint(value: int, form=int_renderer) -> str:
        sv: str = form(value).strip()
        if (sv is not None) and (not isinstance(sv, str)):
            raise type_error(s, f"conversion of {value}", (str, None))
        return sv

    # step four: if all data are integer, we can convert them directly
    if all_is_int:
        # an int render also processing None and special floats
        def __toint2(value: Union[None, int, float],
                     form=__toint,
                     na=nan_str,
                     pi=positive_infty_str,
                     ni=negative_infty_str) -> Optional[str]:
            if value is None:
                return None
            return form(cast(int, value)) if isfinite(value)\
                else na if isnan(value) else pi if value >= inf else ni
        return [__toint2(i) for i in data]

    # ok, we have at least some finite floats that cannot be converted to
    # integers. therefore, we need to convert them to strings based on a
    # floating point number format.
    float_format = float_format_getter(min_finite, max_finite,
                                       longest_fraction)
    if not isinstance(float_format, str):
        raise type_error(float_format,
                         "float format from float_format_getter", str)
    if (len(float_format) <= 0) or ('{' not in float_format) \
            or ('}' not in float_format) or (':' not in float_format):
        raise ValueError(f"invalid float format '{float_format}'.")

    # step five: first, create the raw float strings and mark special values
    result: Final[List[Union[int, str]]] = [
        0 if value is None
        else float_format.format(value).strip() if isfinite(value)
        else 1 if isnan(value)
        else 2 if value >= inf
        else 3 for value in data]

    # step six: fix special values and float strings
    for i, value in enumerate(result):
        if value == 0:
            result[i] = none_str
        elif value == 1:
            result[i] = nan_str
        elif value == 2:
            result[i] = positive_infty_str
        elif value == 3:
            result[i] = negative_infty_str
        elif not isinstance(value, str):
            raise type_error(value, "computed value", str)
        else:
            # if we get here, we have to deal with actual floats.
            # we split them into int, frac, and exp parts.
            # if the int and exp part exist, we format them as ints.
            int_part: str
            frac_part: str = ""
            exp_part: str = ""
            eidx: int = value.find("e")
            if eidx < 0:
                eidx = value.find("E")
            if eidx >= 0:
                exp_part = exponent_renderer(__toint(int(
                    value[eidx + 1:])).strip()).strip()
                if not isinstance(exp_part, str):
                    raise type_error(exp_part,
                                     "exponent part rendering result", str)
                value = value[:eidx].strip()

            dotidx: int = value.find(".")
            if dotidx <= 0:
                int_part = __toint(int(value))
            else:
                int_part = __toint(int(value[:dotidx]))
                frac_part = value[dotidx:].strip()
            if len(int_part) <= 0:
                int_part = "0"
            result[i] = f"{int_part}{frac_part}{exp_part}"

    return cast(List[str], result)
