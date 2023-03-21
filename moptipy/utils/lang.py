"""
The :class:`Lang` class provides facilities for easy internationalization.

The idea here is to have simply tools that provide locale-specific keywords,
texts, and number formats.
"""

from typing import Callable, Final, Iterable

from matplotlib import rc  # type: ignore
from matplotlib.font_manager import (  # type: ignore
    FontProperties,
    findSystemFonts,
)

from moptipy.utils.strings import sanitize_name
from moptipy.utils.types import check_int_range, type_error


class Lang:
    """A language-based dictionary for locale-specific keywords."""

    def __init__(self, name: str, font: str, decimal_stepwidth: int = 3,
                 data: dict[str, str] | None = None,
                 is_default: bool = False):
        """
        Instantiate the language formatter.

        :param name: the short name
        :param font: the font name
        :param decimal_stepwidth: the decimal step width
        :param data: the data
        :param is_default: is this the default language?
        """
        #: the name of the locale
        self.__name: Final[str] = sanitize_name(name)
        if not isinstance(font, str):
            raise type_error(font, "font", str)
        if not isinstance(is_default, bool):
            raise type_error(is_default, "is_default", bool)

        font = font.strip()
        if not font:
            raise ValueError(f"The font cannot be {font!r}.")
        #: the font name
        self.__font: Final[str] = font
        #: the decimal step width
        self.__decimal_stepwidth: Final[int] = check_int_range(
            decimal_stepwidth, "decimal_stepwidth", 1, 10)

        #: the dictionary with the translation data
        self.__dict: Final[dict[str, str]] = {}
        if data:
            self.extend(data)

        #: is this the default language?
        self.__is_default: Final[bool] = is_default

        # register the language
        dc: Final[dict[str, Lang]] = Lang.__get_langs()
        if self.__name in dc:
            raise ValueError(f"Language {self.__name!r} already registered.")
        if is_default:
            for lang in dc.values():
                if lang.__is_default:
                    raise ValueError(
                        f"Language {self.__name!r} cannot be default "
                        f"language, {lang.__name!r} already is!")
        dc[self.__name] = self

        if is_default:
            self.set_current()

    def extend(self, data: dict[str, str]) -> None:
        """
        Add a set of entries to this dictionary.

        :param data: the language-specific data
        """
        if not isinstance(data, dict):
            raise type_error(data, "data", dict)
        for ko, v in data.items():
            k = sanitize_name(ko)
            if k != ko:
                raise ValueError(f"key {ko!r} is different from "
                                 f"its sanitized version {k!r}.")
            if (k in self.__dict) and (self.__dict[k] != v):
                raise ValueError(
                    f"Key {k!r} appears twice, already assigned to "
                    f"{self.__dict[k]!r}, cannot assign to {v!r}.")
            if not isinstance(v, str):
                raise type_error(v, f"value for key {k!r}", str)
            if not v:
                raise ValueError(f"Value for key {k!r} cannot be {v!r}.")
            self.__dict[k] = v

    def filename(self, base: str) -> str:
        """
        Make a suitable filename by appending the language id.

        :param str base: the basename
        :return: the filename
        :rtype: str

        >>> print(Lang.get("en").filename("test"))
        test
        >>> print(Lang.get("zh").filename("test"))
        test_zh
        """
        base = sanitize_name(base)
        return base if self.__is_default else f"{base}_{self.__name}"

    def __repr__(self):
        """
        Get the language's name.

        :return: the language's name
        :rtype: str
        """
        return self.__name

    def __getitem__(self, item: str) -> str:
        """
        Get the language formatting code.

        :param item: the item to get
        :return: the language-specific code
        """
        if not isinstance(item, str):
            raise type_error(item, "item", str)
        return self.__dict[item]

    def format_str(self, item: str, **kwargs) -> str:
        """
        Return a string based on the specified format.

        :param item: the key
        :param kwargs: the keyword-based arguments

        >>> l = Lang.get("en")
        >>> l.extend({"z": "{a}: bla{b}"})
        >>> print(l.format_str("z", a=5, b=6))
        5: bla6
        """
        if not isinstance(item, str):
            raise type_error(item, "item", str)
        fstr: str = self.__dict[item]
        # pylint: disable=W0123 # noqa: DUO104
        return eval(f"f{fstr!r}",  # nosec # nosemgrep # noqa: DUO104,B028
                    {"__builtins__": None},  # nosec # nosemgrep # noqa:DUO104
                    kwargs).strip()  # nosec # nosemgrep # noqa: DUO104

    def font(self) -> str:
        """
        Get the default font for this language.

        :return: the default font for this language

        >>> print(Lang.get("en").font())
        DejaVu Sans
        >>> print(Lang.get("zh").font())
        Noto Sans SC
        """
        return self.__font

    def format_int(self, value: int) -> str:
        """
        Convert an integer to a string.

        :param value: the value
        :returns: a string representation of the value

        >>> print(Lang.get("en").format_int(100000))
        100'000
        >>> print(Lang.get("zh").format_int(100000))
        10'0000
        """
        if not isinstance(value, int):
            raise type_error(value, "value", int)
        if value < 0:
            prefix = "-"
            value = -value
        else:
            prefix = ""

        sss = str(value)
        i = len(sss)
        if i <= self.__decimal_stepwidth:  # no formatting needed
            return prefix + sss

        # We divide the string into equally-sized chunks and insert "'"
        # between them.
        chunks: list[str] = []
        for i in range(i, -1, -self.__decimal_stepwidth):  # noqa
            k: str = sss[i:(i + self.__decimal_stepwidth)]
            if k:
                chunks.insert(0, k)
        if i > 0:
            chunks.insert(0, sss[0:i])
        return prefix + "'".join(chunks)

    @staticmethod
    def __get_langs() -> dict[str, "Lang"]:
        """
        Get the languages decode.

        :return: the languages decode
        """
        att: Final[str] = "__map"
        if not hasattr(Lang.__get_langs, att):
            setattr(Lang.__get_langs, att, {})
        return getattr(Lang.__get_langs, att)

    @staticmethod
    def get(name: str) -> "Lang":
        """
        Get the language of the given key.

        :param name: the language name
        :return: the language
        """
        name = sanitize_name(name)
        lang: Lang | None = Lang.__get_langs().get(name, None)
        if lang:
            return lang
        raise ValueError(f"Unknown language {name!r}.")

    @staticmethod
    def current() -> "Lang":
        """
        Get the current language.

        :return: the current language

        >>> Lang.get("en").set_current()
        >>> print(Lang.current().filename("b"))
        b
        >>> Lang.get("zh").set_current()
        >>> print(Lang.current().filename("b"))
        b_zh
        """
        lang: Final[Lang] = getattr(Lang.__get_langs, "__current")
        if not lang:
            raise ValueError("Huh?")
        return lang

    def set_current(self) -> None:
        """Mark this language as the current one."""
        setattr(Lang.__get_langs, "__current", self)
        rc("font", family=self.font())

    @staticmethod
    def all_langs() -> Iterable["Lang"]:
        """
        Get all presently loaded languages.

        :return: an Iterable of the languages
        """
        val = list(Lang.__get_langs().values())
        val.sort(key=lambda x: x.__name)
        return val

    @staticmethod
    def translate(key: str) -> str:
        """
        Translate the given key to a string in the current language.

        :param key: the key
        :returns: the value of the key in the current language

        >>> EN.extend({'a': 'b'})
        >>> EN.set_current()
        >>> print(Lang.translate("a"))
        b
        >>> DE.extend({'a': 'c'})
        >>> Lang.get("de").set_current()
        >>> print(Lang.translate("a"))
        c
        """
        return Lang.current()[key]

    @staticmethod
    def translate_call(key: str) -> Callable:
        """
        Get a callable that always returns the current translation of a key.

        The callable will ignore all of its parameters and just return the
        translation. This means that you can pass parameters to it and they
        will be ignored.

        :param key: the key to translate
        :returns: the callable doing the translation

        >>> cal = Lang.translate_call("a")
        >>> EN.extend({'a': 'b'})
        >>> EN.set_current()
        >>> print(cal())
        b
        >>> print(cal(1, 2, 3))
        b
        >>> DE.extend({'a': 'c'})
        >>> Lang.get("de").set_current()
        >>> print(cal("x"))
        c
        """
        def __trc(*_, ___key=key) -> str:
            return Lang.current()[___key]
        return __trc

    @staticmethod
    def translate_func(func: str) -> Callable[[str], str]:
        """
        Create a lambda taking a dimensions and presenting a function thereof.

        :param func: the function name
        :returns: the function

        >>> Lang.get("en").set_current()
        >>> Lang.get("en").extend({"ERT": "ERT"})
        >>> Lang.get("en").extend({"FEs": "time in FEs"})
        >>> f = Lang.translate_func("ERT")
        >>> print(f("FEs"))
        ERT\u2009[time in FEs]
        >>> Lang.get("de").set_current()
        >>> Lang.get("de").extend({"ERT": "ERT"})
        >>> Lang.get("de").extend({"FEs": "Zeit in FEs"})
        >>> print(f("FEs"))
        ERT\u2009[Zeit in FEs]
        """
        def __tf(dim: str, f: str = func) -> str:
            return f"{Lang.translate(f)}\u2009[{Lang.translate(dim)}]"
        return __tf


# noinspection PyBroadException
def __get_font(choices: list[str]) -> str:
    """
    Try to find an installed version of the specified font.

    :param choices: the choices of the fonts
    :returns: the installed name of the font, or the value of `choices[0]`
        if no font was found
    """
    if not isinstance(choices, list):
        raise type_error(choices, "choices", list)
    if len(choices) <= 0:
        raise ValueError("no font choices are provided!")
    attr: Final[str] = "fonts_list"
    func: Final = globals()["__get_font"]

    # get the list of installed fonts
    font_list: list[str]
    if not hasattr(func, attr):
        font_list = []
        for fname in findSystemFonts():
            try:
                font_name = FontProperties(
                    fname=fname).get_name().strip()
                if font_name.encode("ascii", "ignore").decode() == font_name:
                    font_list.append(font_name)
            except Exception:  # noqa
                # We can ignore the exceptions here.
                continue  # nosec
        if len(font_list) <= 0:
            raise ValueError("Did not find any font.")
        setattr(func, attr, font_list)
    else:
        font_list = getattr(func, attr)

    # find the installed font
    for choice in choices:
        clc: str = choice.lower()
        found_inside: str | None = None
        found_start: str | None = None
        for got in font_list:
            gotlc = got.lower()
            if clc == gotlc:
                return got
            if gotlc.startswith(clc):
                found_start = got
            if clc in gotlc:
                found_inside = got
        if found_start:
            return found_start
        if found_inside:
            return found_inside

    return choices[0]  # nothing found ... return default


#: the English language
EN: Final[Lang] = Lang("en",
                       __get_font(["DejaVu Sans", "Calibri",
                                   "Arial", "Helvetica"]),
                       3, is_default=True)

#: the German language
DE: Final[Lang] = Lang("de", EN.font(), 3)

#: the Chinese language
ZH: Final[Lang] = Lang("zh",
                       __get_font(["Noto Sans SC", "FangSong", "SimSun",
                                   "Arial Unicode", "SimHei"]),
                       4)

del __get_font  # get rid of no-longer needed data such as fonts list
