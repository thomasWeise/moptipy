"""Styler allows to discover groups of data and associate styles with them."""
from typing import Any, Callable, Final, Iterable, cast

from matplotlib.artist import Artist  # type: ignore
from matplotlib.lines import Line2D  # type: ignore

from moptipy.utils.plot_defaults import create_line_style
from moptipy.utils.types import type_error


class Styler:
    """A class for determining groups of elements and styling them."""

    #: The tuple with the names becomes valid after compilation.
    names: tuple[str, ...]
    #: The tuple with the keys becomes valid after compilation.
    keys: tuple[Any, ...]
    #: The dictionary mapping keys to indices; only valid after compilation.
    __indexes: dict[Any, int]
    #: Is there a None key? Valid after compilation.
    has_none: bool
    #: The number of registered keys.
    count: int
    #: Does this styler have any style associated with it?
    has_style: bool

    def __init__(self,
                 key_func: Callable = lambda x: x,
                 namer: Callable[[Any], str] = str,
                 none_name: str = "None",
                 priority: int | float = 0,
                 name_sort_function: Callable[[str], Any] | None =
                 lambda s: s):
        """
        Initialize the style grouper.

        :param key_func: the key function, obtaining keys from objects
        :param namer: the name function, turning keys into names
        :param none_name: the name for the none-key
        :param priority: the base priority of this grouper
        :param name_sort_function: the function for sorting names, or `None`
            if no name-based sorting shall be performed
        """
        if not callable(key_func):
            raise type_error(key_func, "key function", call=True)

        if not callable(namer):
            raise type_error(namer, "namer function", call=True)
        if not isinstance(none_name, str):
            raise type_error(none_name, "none_name", str)
        if not isinstance(priority, float | int):
            raise type_error(priority, "priority", (int, float))
        if (name_sort_function is not None) \
                and (not callable(name_sort_function)):
            raise type_error(name_sort_function, "name_sort_function",
                             type(None), call=True)

        def __namer(key,
                    __namer: Callable[[Any], str] = namer,
                    __none_name: str = none_name) -> str:
            rv = __none_name if key is None else __namer(key)
            if not isinstance(rv, str):
                raise type_error(rv, f"name for key {key!r}", str)
            rv = rv.strip()
            if len(rv) <= 0:
                raise ValueError(
                    "name cannot be empty or just consist of white space")
            return rv

        #: the name sort function
        self.__name_sort_function: Final[Callable[[str], Any] | None] = \
            name_sort_function
        #: The key function of the grouper
        self.key_func: Final[Callable] = key_func
        #: The name function of the grouper
        self.name_func: Final[Callable[[Any], str]] = \
            cast(Callable[[Any], str], __namer)
        #: The base priority of this grouper
        self.priority: float = float(priority)
        #: The internal collection.
        self.__collection: set = set()
        #: the line colors
        self.__line_colors: tuple | None = None
        #: the line dashes
        self.__line_dashes: tuple | None = None
        #: the line widths
        self.__line_widths: tuple[float, ...] | None = None
        #: the optional line alpha
        self.__line_alphas: tuple[float, ...] | None = None

    def add(self, obj) -> None:
        """
        Add an object to the style collection.

        :param obj: the object
        """
        self.__collection.add(self.key_func(obj))

    def finalize(self) -> None:
        """Compile the styler collection."""
        self.has_none = (None in self.__collection)
        if self.has_none:
            self.__collection.remove(None)

        nsf: Final[Callable[[str], Any] | None] = self.__name_sort_function
        if nsf is None:
            data = [(k, self.name_func(k)) for k in self.__collection]
            data.sort()
            if self.has_none:
                data.insert(0, (None, self.name_func(None)))
            self.keys = tuple([x[0] for x in data])
            self.names = tuple([x[1] for x in data])
        else:
            data = [(self.name_func(k), k) for k in self.__collection]
            data.sort(key=cast(Callable, lambda x, nsf2=nsf: nsf2(x[0])))
            if self.has_none:
                data.insert(0, (self.name_func(None), None))
            self.names = tuple([x[0] for x in data])
            self.keys = tuple([x[1] for x in data])

        del self.__collection
        del data
        self.__indexes = {k: i for i, k in enumerate(self.keys)}
        self.count = len(self.names)
        self.priority += self.count
        self.has_style = False

    def __lt__(self, other) -> bool:
        """
        Check whether this styler is more important than another one.

        :param other: the other styler
        :return: `True` if it is, `False` if it is not.
        """
        if self.priority > other.priority:
            return True
        if self.priority < other.priority:
            return False
        c1 = self.count
        if self.has_none:
            c1 -= 1
        c2 = other.count
        if other.has_none:
            c2 -= 1
        if c1 > c2:
            return True
        return False

    def set_line_color(self, line_color_func: Callable) -> None:
        """
        Set that this styler should apply line colors.

        :param line_color_func: a function returning the palette
        """
        tmp = line_color_func(self.count)
        if not isinstance(tmp, Iterable):
            raise type_error(tmp, "result of line color func", Iterable)
        self.__line_colors = tuple(tmp)
        if len(self.__line_colors) != self.count:
            raise ValueError(f"There must be {self.count} line colors,"
                             f"but found only {len(self.__line_colors)}.")
        self.has_style = True

    def set_line_dash(self, line_dash_func: Callable) -> None:
        """
        Set that this styler should apply line dashes.

        :param line_dash_func: a function returning the dashes
        """
        tmp = line_dash_func(self.count)
        if not isinstance(tmp, Iterable):
            raise type_error(tmp, "result of line dash func", Iterable)
        self.__line_dashes = tuple(tmp)
        if len(self.__line_dashes) != self.count:
            raise ValueError(f"There must be {self.count} line dashes,"
                             f"but found only {len(self.__line_dashes)}.")
        self.has_style = True

    def set_line_width(self, line_width_func: Callable) -> None:
        """
        Set that this styler should apply a line width.

        :param line_width_func: the line width function
        """
        tmp = line_width_func(self.count)
        if not isinstance(tmp, Iterable):
            raise type_error(tmp, "result of line width func", Iterable)
        self.__line_widths = tuple(tmp)
        if len(self.__line_widths) != self.count:
            raise ValueError(f"There must be {self.count} line widths,"
                             f"but found only {len(self.__line_widths)}.")
        self.has_style = True

    def set_line_alpha(self, line_alpha_func: Callable) -> None:
        """
        Set that this styler should apply a line alpha.

        :param line_alpha_func: the line alpha function
        """
        tmp = line_alpha_func(self.count)
        if not isinstance(tmp, Iterable):
            raise type_error(tmp, "result of line alpha func", Iterable)
        self.__line_alphas = tuple(tmp)
        if len(self.__line_alphas) != self.count:
            raise ValueError(f"There must be {self.count} line alphas,"
                             f"but found only {len(self.__line_alphas)}.")
        self.has_style = True

    def add_line_style(self, obj,
                       style: dict[str, object]) -> None:
        """
        Apply this styler's contents based on the given object.

        :param obj: the object for which the style should be created
        :param style: the decode to which the styles should be added
        """
        key = self.key_func(obj)
        index = self.__indexes.setdefault(key, -1)
        if index >= 0:
            self.__add_line_style(index, style)

    def __add_line_style(self, index,
                         style: dict[str, object]) -> None:
        """
        Apply this styler's contents based on the given object.

        :param index: the index to be processed
        :param style: the decode to which the styles should be added
        """
        if self.__line_colors is not None:
            style["color"] = self.__line_colors[index]
        if self.__line_dashes is not None:
            style["linestyle"] = self.__line_dashes[index]
        if self.__line_widths is not None:
            style["linewidth"] = self.__line_widths[index]
        if self.__line_alphas is not None:
            style["alpha"] = self.__line_alphas[index]

    def add_to_legend(self, consumer: Callable[[Artist], Any]) -> None:
        """
        Add this styler to the legend.

        :param consumer: the consumer to add to
        """
        if not callable(consumer):
            raise type_error(consumer, "consumer", call=True)
        for i, name in enumerate(self.names):
            style = create_line_style()
            self.__add_line_style(i, style)
            style["label"] = name
            style["xdata"] = []
            style["ydata"] = []
            consumer(Line2D(**style))
