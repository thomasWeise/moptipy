"""Utilities for creating visualizations in multiple languages."""

from typing import Dict, Final, Optional, Iterable, List

import matplotlib  # type: ignore

from moptipy.utils.logging import sanitize_name


class Lang:
    """A language-based dictionary for locale-specific keywords."""

    def __init__(self,
                 name: str,
                 font: str,
                 decimal_stepwidth: int,
                 data: Dict[str, str]):
        """
        Instantiate the language formatter.

        :param str name: the short name
        :param str font: the font name
        :param int decimal_stepwidth: the decimal step width
        :param Dict[str, str] data: the data
        """
        #: the name of the locale
        self.__name: Final[str] = sanitize_name(name)

        if not isinstance(font, str):
            raise TypeError(f"The font must be str, but is {type(font)}.")
        font = font.strip()
        if not font:
            raise ValueError(f"The font cannot be '{font}'.")
        #: the font name
        self.__font: Final[str] = font

        if not isinstance(decimal_stepwidth, int):
            raise TypeError(f"The decimal stepwidth must be int, but "
                            f"is {type(decimal_stepwidth)}.")
        if decimal_stepwidth <= 1:
            raise TypeError(f"The decimal stepwidth must be > 1, but "
                            f"is {decimal_stepwidth}.")
        #: the decimal step width
        self.__decimal_stepwidth: Final[int] = decimal_stepwidth

        #: the dictionary with the translation data
        self.__dict: Final[Dict[str, str]] = {}
        self.extend(data)

    def extend(self, data: Dict[str, str]) -> None:
        """
        Add a set of entries to this dictionary.

        :param Dict[str, str] data: the language-specific data
        """
        if not isinstance(data, dict):
            raise TypeError(f"Data must be Dict, but is {type(data)}.")
        for k, v in data.items():
            k = sanitize_name(k)
            if k in self.__dict:
                raise ValueError(
                    f"Key '{k}' appears twice, already assigned to "
                    f"'{self.__dict[k]}', cannot assign to '{v}'.")
            if not isinstance(v, str):
                raise TypeError(f"Value for key '{k}' must be str, "
                                f"but is {type(v)}.")
            if not v:
                raise ValueError(f"Value for key '{k}' cannot be '{v}'.")
            self.__dict[k] = v

    def filename(self, base: str) -> str:
        """
        Make a suitable filename by appending the language id.

        :param str base: the basename
        :return: the filename
        :rtype: str
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

        :param str item: the item to get
        :return: the language-specific code
        :rtype: str
        """
        if not isinstance(item, str):
            raise TypeError(f"Item must be str, but is {type(item)}.")
        return self.__dict[item]

    def format(self, item: str, **kwargs):
        """
        Return a string based on the specified format.

        :param str item: the key
        :param kwargs: the keyword-based arguments
        :rtype: str
        """
        if not isinstance(item, str):
            raise TypeError(f"Item must be str, but is {type(item)}.")
        fstr = self.__dict[item]
        # pylint: disable=W0123
        return eval(f'f"""{fstr}"""',  # nosec
                    {"__builtins__": None}, kwargs).strip()  # nosec

    def font(self) -> str:
        """
        Get the default font for this language.

        :return: the default font for this language
        :rtype: str
        """
        return self.__font

    def format_int(self, value: int) -> str:
        """
        Convert an integer to a string.

        :param int value: the value
        :returns: a string representation of the value
        :rtype: str
        """
        if not isinstance(value, int):
            raise TypeError(f"Value must be int, but is {type(value)}.")
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
        :rtype: Dict[str, Lang]
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

        :param str name: the language name
        :return: the language
        :rtype: the type
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
        :rtype: Lang
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
        :rtype: Iterable['Lang']
        """
        val = list(Lang.__get_langs().values())
        val.sort(key=lambda x: x.__name)
        return val


lang_en = Lang("en", "DejaVu Sans", 3, {
    "f": "f",
    "feasible": "feasible",
    "name": "name",
    "time": "time",
    "time_in_fes": "time in FEs",
    "time_in_ms": "time in ms",
})
lang_en.register()
lang_en.set_current()

Lang("de", lang_en.font(), 3, {
    "f": "f",
    "feasible": "realisierbar",
    "name": "Name",
    "time": "Zeit",
    "time_in_ms": "Zeit in ms",
    "time_in_fes": "Zeit in FEs",
}).register()

del lang_en

Lang("zh", "Noto Sans SC", 4, {
    "f": "f",
    "feasible": "可行的",
    "name": "名称",
    "time": "时间",
    "time_in_ms": "时间(毫秒)",
    "time_in_fes": "时间(目标函数的评价)",
}).register()