"""The numeric format definitions."""

from math import inf, isfinite
from typing import Callable, Final, Iterable, cast

from moptipy.utils.formatted_string import FormattedStr
from moptipy.utils.lang import Lang
from moptipy.utils.math import try_int
from moptipy.utils.types import type_error


def default_get_int_renderer() -> Callable[[int], str]:
    """
    Get the default integer renderer.

    :returns: the default integer renderer, which uses the integer rendering
        of the currently active language setting.

    >>> from moptipy.utils.lang import EN, ZH
    >>> EN.set_current()
    >>> f = default_get_int_renderer()
    >>> f(1_000_000)
    "1'000'000"
    >>> ZH.set_current()
    >>> f = default_get_int_renderer()
    >>> f(1_000_000)
    "100'0000"
    """
    return cast(Callable[[int], str], Lang.current().format_int)


def default_get_float_format(
        min_finite: int | float = 0,
        max_finite: int | float = 0,
        max_frac_len: int = 2,
        min_non_zero_abs: int | float = inf,
        int_to_float_threshold: int | float = 10_000_000_000) -> str:
    """
    Get the default float format for numbers in the given range.

    :param min_finite: the minimum finite value that may need to be formatted
    :param max_finite: the maximum finite value that may need to be formatted
    :param max_frac_len: the length of the longest fractional part of any
        number encountered that can be converted to a string *not* in the "E"
        notation
    :param min_non_zero_abs: the minimum non-zero absolute value; will be
        `inf` if all absolute values are zero
    :param int_to_float_threshold: the threshold above which all integers are
        converted to floating point numbers with the 'E' notation

    >>> default_get_float_format(0, 0, 0)
    '{:.0f}'
    >>> default_get_float_format(0, 1e5, 10)
    '{:.0f}'
    >>> default_get_float_format(-1e7, 1e2, 10)
    '{:.0f}'
    >>> default_get_float_format(0, 0, 1)
    '{:.1f}'
    >>> default_get_float_format(0, 1e3, 11)
    '{:.1f}'
    >>> default_get_float_format(-1e3, 1e2, 11)
    '{:.1f}'
    >>> default_get_float_format(0, 0, 2)
    '{:.2f}'
    >>> default_get_float_format(0, 0, 3)
    '{:.3f}'
    >>> default_get_float_format(0, 0, 4)
    '{:.3f}'
    >>> default_get_float_format(0, 1e11, 4)
    '{:.2e}'
    >>> default_get_float_format(-1, 1, 4, 1e-3)
    '{:.3f}'
    """
    if not isinstance(min_finite, (int, float)):
        raise type_error(min_finite, "min_finite", (int, float))
    if not isinstance(max_finite, (int, float)):
        raise type_error(max_finite, "max_finite", (int, float))
    if not (isfinite(min_finite) and isfinite(max_finite)
            and (min_finite <= max_finite)):
        raise ValueError("invalid min_finite, max_finite pair "
                         f"{min_finite}, {max_finite}.")
    if not isinstance(max_frac_len, int):
        raise type_error(max_frac_len, "frac_len", int)
    if not (0 <= max_frac_len < 100):
        raise ValueError(f"invalid frac_len {max_frac_len}.")
    if not isinstance(int_to_float_threshold, (int, float)):
        raise type_error(
            int_to_float_threshold, "int_to_float_threshold", (int, float))
    if (int_to_float_threshold <= 0) or not (
            isfinite(int_to_float_threshold)
            or (int_to_float_threshold >= inf)):
        raise ValueError(
            f"invalid int_to_float_threshold={int_to_float_threshold}.")
    if not isinstance(min_non_zero_abs, (int, float)):
        raise type_error(min_non_zero_abs, "min_non_zero_abs", (int, float))
    if min_non_zero_abs <= 0:
        raise ValueError(f"invalid min_non_zero_abs={min_non_zero_abs}")

    # are the values in the [-1, 1] range, i.e., possibly just small fractions?
    if (min_finite >= -1) and (max_finite <= 1) and isfinite(min_non_zero_abs):
        if min_non_zero_abs >= 1e-1:
            return "{:.1f}"
        if min_non_zero_abs >= 1e-2:
            return "{:.2f}"
        if min_non_zero_abs >= 1e-3:
            return "{:.3f}"
        if min_non_zero_abs >= 1e-4:
            return "{:.4f}"
        return "{:.3e}"

    # handle numbers that are outside [-1, 1]
    if ((-int_to_float_threshold) <= min_finite) \
            and (max_finite <= int_to_float_threshold):
        if (max_frac_len <= 0) or (min_finite <= -1E4) or (max_finite >= 1E4):
            return "{:.0f}"
        if (max_frac_len <= 1) or (min_finite <= -1E3) or (max_finite >= 1E3):
            return "{:.1f}"
        if (max_frac_len <= 2) or (min_finite <= -1E2) or (max_finite >= 1E2):
            return "{:.2f}"
        return "{:.3f}"
    return "{:.2e}"


class NumberRenderer:
    """
    A format description for a group of numbers.

    With instances of this class, you can convert a sequence of numbers
    to a sequence of strings with uniform, pleasant formatting. The idea
    is that such numbers can be written, e.g., into a column of a table
    and that this column will then have a nice and uniform appearance.
    In other words, we will avoid situations like the following:
    "1234938845, 1e-20, 0.002, 34757773, 1e30, 0.9998837467"
    which looks rather odd. While the numbers may be displayed correctly,
    the formatting of all numbers is different. If we want to present
    numbers that describe related quantities, we rather want them to all
    have the same format. This class here can achieve this in a customizable
    way.
    """

    def __init__(self,
                 int_to_float_threshold: int | float = 10_000_000_000,
                 get_int_renderer: Callable[[], Callable[[int], str]]
                 = default_get_int_renderer,
                 get_float_format: Callable[
                     [int | float, int | float, int,
                      int | float, int | float], str] =
                 default_get_float_format):
        """
        Create the number group format.

        :param int_to_float_threshold: the threshold above which integers are
            converted to floating point numbers in the 'E' notation.
        :param get_int_renderer: the function to be used to get the renderer
            for all integers and integer parts of floats.
        :param get_float_format: the getter for the float format, i.e., a
            callable accepting the range [min, max] of all finite values to be
            rendered, the maximum length of any fractional part, and the
            `int_to_float_threshold` value and that then returns a string with
            the float format definition
        """
        super().__init__()
        while True:
            if not isinstance(int_to_float_threshold, (int, float)):
                raise type_error(int_to_float_threshold,
                                 "int_to_float_threshold", (int, float))
            if (int_to_float_threshold <= 0) or not (
                    isfinite(int_to_float_threshold)
                    or (int_to_float_threshold >= inf)):
                raise ValueError("invalid int_to_float_threshold="
                                 f"{int_to_float_threshold}.")
            if isinstance(int_to_float_threshold, float):
                a = int(int_to_float_threshold)
                if a == int_to_float_threshold:
                    int_to_float_threshold = a
            else:
                break
        #: the absolute threshold above which all integer numbers must be
        #: converted to floats to render them in the 'E' notation
        self.int_to_float_threshold: Final[int | float] \
            = int_to_float_threshold
        if not callable(get_int_renderer):
            raise type_error(get_int_renderer, "int_renderer", call=True)
        #: the function to be used to get the renderer for all integers and
        #: integer parts of floats
        self.get_int_renderer: Final[Callable[[], Callable[[int], str]]] \
            = get_int_renderer
        #: the getter for the float format to be used to represent a range of
        #: values
        self.get_float_format: Final[Callable[
            [int | float, int | float, int,
             int | float, int | float], str]] = get_float_format

    def derive(self,
               int_to_float_threshold: int | float | None = None,
               get_int_renderer: Callable[[], Callable[
                   [int], str]] | None = None,
               get_float_format: Callable[[int | float, int | float, int,
                                           int | float, int | float],
                                          str] | None = None) \
            -> "NumberRenderer":
        """
        Derive a new number group format from this one.

        :param int_to_float_threshold: the int-to-float threshold
        :param get_int_renderer: the integer renderer getter
        :param get_float_format: the float format getter
        :returns: a new number group format that differs from the current
            format only in terms of the non-`None` parameters specified

        >>> d = DEFAULT_NUMBER_RENDERER
        >>> d.derive() is d
        True
        >>> d.int_to_float_threshold
        10000000000
        >>> from moptipy.utils.lang import EN
        >>> EN.set_current()
        >>> d.get_int_renderer()(123456789)
        "123'456'789"
        >>> d.get_float_format(-10, 10, 2)
        '{:.2f}'
        >>> d = d.derive(int_to_float_threshold=22)
        >>> d is DEFAULT_NUMBER_RENDERER
        False
        >>> d.int_to_float_threshold
        22
        >>> d = d.derive(get_int_renderer=lambda: lambda x: "bla")
        >>> d.get_int_renderer()(112)
        'bla'
        """
        # pylint: disable=R0916
        if (((int_to_float_threshold is None)
             or (int_to_float_threshold == self.int_to_float_threshold))
                and ((get_int_renderer is None)
                     or (get_int_renderer is self.get_int_renderer))
                and ((get_float_format is None)
                     or (get_float_format is self.get_float_format))):
            return self
        return NumberRenderer(
            self.int_to_float_threshold if
            int_to_float_threshold is None else int_to_float_threshold,
            self.get_int_renderer if get_int_renderer is None
            else get_int_renderer,
            self.get_float_format if get_float_format is None
            else get_float_format)

    def render(self, source: int | float | None | Iterable[int | float | None],
               none_str: FormattedStr | None = None) \
            -> list[FormattedStr | None]:
        r"""
        Convert a sequence of numbers to text with uniform shape.

        Often, we need to convert a set of numbers to strings as output for a
        table or another representative thext. In such a case, you want to
        present all numbers in the set in the same format.

        Imagine you have the number vector `[1E-4, 1/7, 123456789012345678]`.
        If you simply convert this list to a string directly, what you get is
        `[0.0001, 0.14285714285714285, 123456789012345678]`. Now this looks
        very ugly. First, we have one very big number `123456789012345678`.
        If the numbers stem from an experiment, then we are hardly able to
        obtain any number at a very extreme precision. The 18 digits in
        `123456789012345678` sort of suggest a precision to 18 decimals, since
        the number ends in specific digits (as opposed to `123450000000000000`
        which a reader would naturally preceive as a rounded quantity).
        Additionally, we have the number `0.14285714285714285`, which has a
        very long fractional part, which, too, suggests a very high precision.
        Writing both mentioned numbers next to each other, this suggests as if
        we could present a number as high as 10**18 at a precision of 10**-17.
        And it also looks ugly, because both numbers are not uniformly
        formatted. Instead, our function here renders the number list as
        `['1.00*10^-4^', '1.43*10^-1^', '1.23*10^17^']`. It recognizes that we
        should present numbers as powers of ten and then limits the precision
        to three digits.

        This function is thus intended to produce some sort of uniform format
        with reasonable precision uniformly for a numerical vector, under the
        assumption that all numbers should be presented in the same numerical
        range and quantity.

        :param source: the column data
        :param none_str: the string replacement for `None`
        :returns: a list with the text representation

        >>> from moptipy.utils.lang import EN
        >>> EN.set_current()
        >>> ff = DEFAULT_NUMBER_RENDERER
        >>> ff.render([1.75651, 212, 3234234])
        ['2', '212', "3'234'234"]
        >>> ff.render([1.75651, 22, 34])
        ['1.757', '22.000', '34.000']
        >>> ff.render([1.75651, 122, 34])
        ['1.76', '122.00', '34.00']
        >>> ff.render([1.75651, 122, 3334])
        ['1.8', '122.0', "3'334.0"]
        >>> ff.render([1.5, 212, 3234234])
        ['2', '212', "3'234'234"]
        >>> ff.render([1.5, 2e12, 3234234])
        ['1.50e0', '2.00e12', '3.23e6']
        >>> ff.render([233, 22139283482834, 3234234])
        ['2.33e2', '2.21e13', '3.23e6']
        >>> ff.render([233, 22139283, 3234234])
        ['233', "22'139'283", "3'234'234"]
        >>> from math import nan, inf
        >>> ff.render([22139283, inf, -inf, nan, None])
        ["22'139'283", 'inf', '-inf', 'nan', None]
        >>> ff.render([1E-4, 1/7, 123456789012345678])
        ['1.00e-4', '1.43e-1', '1.23e17']
        >>> ff.render([0, 0.02, 0.1, 1e-3])
        ['0.000', '0.020', '0.100', '0.001']
        >>> ff.render([-0.2, 1e-6, 0.9])
        ['-2.000e-1', '1.000e-6', '9.000e-1']
        """
        if (source is None) or isinstance(source, (int, float)):
            source = [source]
        if not isinstance(source, Iterable):
            raise type_error(source, "source", Iterable)
        if (none_str is not None) and (
                not isinstance(none_str, FormattedStr)):
            raise type_error(none_str, "none_str", (str, None))

        # get the format parameters
        int_renderer: Final[Callable[[int], str]] = \
            self.get_int_renderer()
        if not callable(int_renderer):
            raise type_error(int_renderer, "int_renderer", call=True)
        int_to_float_threshold: Final[int | float] \
            = self.int_to_float_threshold

        # step one: get the raw numerical data
        data: Final[list[int | float | None]] = \
            cast(list, source) if isinstance(source, list) else list(source)
        dlen: Final[int] = len(data)
        if dlen <= 0:
            raise ValueError("Data cannot be empty.")

        # step two: investigate the data ranges and structure
        all_is_none: bool = True
        all_is_int: bool = True
        max_finite: int | float = -inf
        min_finite: int | float = inf
        min_non_zero_abs: int | float = inf
        longest_fraction: int = -1
        da: int | float

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
                da = abs(d2)
                if 0 < da < min_non_zero_abs:
                    min_non_zero_abs = da
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
                da = abs(d2)
                if 0 < da < min_non_zero_abs:
                    min_non_zero_abs = da

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
            def __toint2(value: None | int | float, _ns=none_str,
                         form=__toint) -> FormattedStr | None:
                if value is None:
                    return none_str
                return FormattedStr.number(form(cast(int, value))
                                           if isfinite(value) else value)
            return [__toint2(i) for i in data]

        # ok, we have at least some finite floats that cannot be converted to
        # integers. therefore, we need to convert them to strings based on a
        # floating point number format.
        float_format = self.get_float_format(
            min_finite, max_finite, longest_fraction, min_non_zero_abs,
            int_to_float_threshold)
        if not isinstance(float_format, str):
            raise type_error(float_format,
                             "float format from float_format_getter", str)
        if (len(float_format) <= 0) or ("{" not in float_format) \
                or ("}" not in float_format) or (":" not in float_format):
            raise ValueError(f"invalid float format '{float_format}'.")

        def __render_float(value: int | float, ir=__toint,
                           ff=float_format) -> FormattedStr:
            if value is None:
                return none_str
            if isfinite(value):
                res: str = ff.format(value).strip()
                int_part: str
                frac_part: str = ""
                exp_part: str = ""
                eidx: int = res.find("e")
                if eidx < 0:
                    eidx = res.find("E")
                if eidx >= 0:
                    exp_part = f"e{ir(int(res[eidx + 1:])).strip()}"
                    res = res[:eidx].strip()
                dotidx: int = res.find(".")
                if dotidx <= 0:
                    int_part = ir(int(res))
                else:
                    int_part = ir(int(res[:dotidx]))
                    frac_part = res[dotidx:].strip()
                if len(int_part) <= 0:
                    int_part = "0"
                return FormattedStr.number(f"{int_part}{frac_part}{exp_part}")
            return FormattedStr.number(value)

        # step five: first, create the raw float strings and mark special
        # values
        return [__render_float(value) for value in data]


#: the default shared singleton instance of the number group format
DEFAULT_NUMBER_RENDERER: Final[NumberRenderer] = NumberRenderer()
