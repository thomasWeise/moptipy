"""
The :class:`Lang` class provides facilities for easy internationalization.

The idea here is to have simply tools that provide locale-specific keywords,
texts, and number formats.
"""

from typing import Dict, Final, Optional, Iterable, List, Callable

import matplotlib  # type: ignore

from moptipy.utils.strings import sanitize_name
from moptipy.utils.types import type_error


class Lang:
    """A language-based dictionary for locale-specific keywords."""

    def __init__(self, name: str, font: str, decimal_stepwidth: int = 3,
                 data: Optional[Dict[str, str]] = None):
        """
        Instantiate the language formatter.

        :param name: the short name
        :param font: the font name
        :param decimal_stepwidth: the decimal step width
        :param data: the data
        """
        #: the name of the locale
        self.__name: Final[str] = sanitize_name(name)

        if not isinstance(font, str):
            raise type_error(font, "font", str)
        font = font.strip()
        if not font:
            raise ValueError(f"The font cannot be '{font}'.")
        #: the font name
        self.__font: Final[str] = font

        if not isinstance(decimal_stepwidth, int):
            raise type_error(decimal_stepwidth, "decimal_stepwidth", int)
        if decimal_stepwidth <= 1:
            raise ValueError(f"The decimal stepwidth must be > 1, but "
                             f"is {decimal_stepwidth}.")
        #: the decimal step width
        self.__decimal_stepwidth: Final[int] = decimal_stepwidth

        #: the dictionary with the translation data
        self.__dict: Final[Dict[str, str]] = {}
        if data:
            self.extend(data)

    def extend(self, data: Dict[str, str]) -> None:
        """
        Add a set of entries to this dictionary.

        :param data: the language-specific data
        """
        if not isinstance(data, dict):
            raise type_error(data, "data", Dict)
        for k, v in data.items():
            k = sanitize_name(k)
            if (k in self.__dict) and (self.__dict[k] != v):
                raise ValueError(
                    f"Key '{k}' appears twice, already assigned to "
                    f"'{self.__dict[k]}', cannot assign to '{v}'.")
            if not isinstance(v, str):
                raise type_error(v, f"value for key '{k}'", str)
            if not v:
                raise ValueError(f"Value for key '{k}' cannot be '{v}'.")
            self.__dict[k] = v

    def filename(self, base: str) -> str:
        """
        Make a suitable filename by appending the language id.

        :param str base: the basename
        :return: the filename
        :rtype: str

        >>> print(Lang.get("en").filename("test"))
        test_en
        >>> print(Lang.get("zh").filename("test"))
        test_zh
        """
        return f"{sanitize_name(base)}_{self.__name}"

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

    def format(self, item: str, **kwargs):
        """
        Return a string based on the specified format.

        :param item: the key
        :param kwargs: the keyword-based arguments

        >>> l = Lang.get("en")
        >>> l.extend({"z": "{a}: bla{b}"})
        >>> print(l.format("z", a=5, b=6))
        5: bla6
        """
        if not isinstance(item, str):
            raise type_error(item, "item", str)
        fstr = self.__dict[item]
        # pylint: disable=W0123
        return eval(f'f"""{fstr}"""',  # nosec # nosemgrep
                    {"__builtins__": None},  # nosec # nosemgrep
                    kwargs).strip()  # nosec # nosemgrep

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
        chunks: List[str] = []
        for i in range(i, -1, -self.__decimal_stepwidth):
            k: str = sss[i:(i + self.__decimal_stepwidth)]
            if k:
                chunks.insert(0, k)
        if i > 0:
            chunks.insert(0, sss[0:i])
        return prefix + "'".join(chunks)

    @staticmethod
    def __get_langs() -> Dict[str, 'Lang']:
        """
        Get the languages map.

        :return: the languages map
        """
        att: Final[str] = "__map"
        if not hasattr(Lang.__get_langs, att):
            setattr(Lang.__get_langs, att, {})
        return getattr(Lang.__get_langs, att)

    def register(self) -> None:
        """Register this language setting."""
        dc: Final[Dict[str, Lang]] = Lang.__get_langs()
        if self.__name in dc:
            raise ValueError(f"Language '{self.__name}' already registered.")
        dc[self.__name] = self

    @staticmethod
    def get(name: str) -> 'Lang':
        """
        Get the language of the given key.

        :param name: the language name
        :return: the language
        """
        name = sanitize_name(name)
        lang: Optional[Lang] = Lang.__get_langs().get(name, None)
        if lang:
            return lang
        raise ValueError(f"Unknown language '{name}'.")

    @staticmethod
    def current() -> 'Lang':
        """
        Get the current language.

        :return: the current language

        >>> Lang.get("en").set_current()
        >>> print(Lang.current().filename("b"))
        b_en
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
        matplotlib.rc("font", family=self.font())

    @staticmethod
    def all() -> Iterable['Lang']:
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
    def translate_func(func: str) -> Callable:
        """
        Create a lambda taking a dimensions and presenting a function thereof.

        :param func: the function name
        :returns: the function

        >>> Lang.get("en").set_current()
        >>> Lang.get("en").extend({"ERT": "ERT"})
        >>> Lang.get("en").extend({"FEs": "time in FEs"})
        >>> f = Lang.translate_func("ERT")
        >>> print(f("FEs"))
        ERT [time in FEs]
        >>> Lang.get("de").set_current()
        >>> Lang.get("de").extend({"ERT": "ERT"})
        >>> Lang.get("de").extend({"FEs": "Zeit in FEs"})
        >>> print(f("FEs"))
        ERT [Zeit in FEs]
        """
        def __tf(dim: str, f: str = func) -> str:
            return f"{Lang.translate(f)}\u2009[{Lang.translate(dim)}]"
        return __tf


#: the English language
EN: Final[Lang] = Lang("en", "DejaVu Sans", 3)
EN.register()
EN.set_current()

#: the German language
DE: Final[Lang] = Lang("de", EN.font(), 3)
DE.register()

#: the Chinese language
ZH: Final[Lang] = Lang("zh", "Noto Sans SC", 4)
ZH.register()
