"""Styler allows to discover groups of data and associate styles with them."""
from typing import Callable, Final, Union, Set, Tuple, Any, Dict, Optional, \
    MutableSequence, Iterable

from matplotlib.artist import Artist  # type: ignore
from matplotlib.lines import Line2D  # type: ignore

from moptipy.evaluation.plot_defaults import create_line_style


class Styler:
    """A class for determining groups of elements and styling them."""

    #: The tuple with the names becomes valid after compilation.
    names: Tuple[str, ...]
    #: The tuple with the keys becomes valid after compilation.
    keys: Tuple[Any, ...]
    #: The dictionary mapping keys to indices; only valid after compilation.
    __indexes: Dict[Any, int]
    #: Is there a None key? Valid after compilation.
    has_none: bool
    #: The number of registered keys.
    count: int
    #: Does this styler have any style associated with it?
    has_style: bool

    def __init__(self,
                 key_func: Callable,
                 namer: Union[str, Callable] = lambda x: x,
                 priority: Union[int, float] = 0):
        """
        Initialize the style grouper.

        :param Callable key_func: the key function, obtaining keys from
            objects
        :param Union[str, Callable] namer: the name function, turning
            keys into names, or the string to be used for replacing `None`
            keys as names (in which case all non-`None` keys are used as
            names as-is)
        :param Union[int, float] priority: the base priority of this grouper
        """
        if not callable(key_func):
            raise TypeError("Key function must be callable.")
        if isinstance(namer, str):
            def name_f(x, nn=namer):
                return nn if x is None else str(x)
        else:
            if not callable(namer):
                raise TypeError(
                    f"Name function must be callable but is {type(namer)}.")
            name_f = namer
        if not isinstance(priority, (float, int)):
            raise TypeError("priority must be float or int "
                            f"but is {type(priority)}.")

        #: The key function of the grouper
        self.key_func: Final[Callable] = key_func
        #: The name function of the grouper
        self.name_func: Final[Callable] = name_f
        #: The base priority of this grouper
        self.priority: float = float(priority)
        #: The internal collection.
        self.__collection: Set = set()
        #: the colors
        self.__line_colors: Optional[Tuple] = None
        #: the line dashes
        self.__line_dashes: Optional[Tuple] = None
        #: the line widths
        self.__line_widths: Optional[Tuple[float, ...]] = None
        #: the optional line alpha
        self.__line_alphas: Optional[Tuple[float, ...]] = None

    def add(self, obj) -> None:
        """
        Add an object to the style collection.

        :param obj: the object
        """
        self.__collection.add(self.key_func(obj))

    def compile(self) -> None:
        """Compile the styler collection."""
        self.has_none = (None in self.__collection)
        if self.has_none:
            self.__collection.remove(None)

        data = [(self.name_func(k), k) for k in self.__collection]
        del self.__collection
        data.sort()
        if self.has_none:
            data.insert(0, (self.name_func(None), None))

        self.names = tuple([x[0] for x in data])
        self.keys = tuple([x[1] for x in data])
        self.__indexes = {k: i for i, k in enumerate(self.keys)}
        self.count = len(self.names)
        self.priority += self.count
        self.has_style = False

    def __lt__(self, other) -> bool:
        """
        Check whether this styler is more important than another one.

        :param other: the other styler
        :return: `True` if it is, `False` if it is not.
        :rtype: bool
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

        :param Callable line_color_func: a function returning the palette
        """
        tmp = line_color_func(self.count)
        if not isinstance(tmp, Iterable):
            raise TypeError(
                f"Line colors must be iterable, but are {type(tmp)}.")
        self.__line_colors = tuple(tmp)
        if len(self.__line_colors) != self.count:
            raise ValueError(f"There must be {self.count} line colors,"
                             f"but found only {len(self.__line_colors)}.")
        self.has_style = True

    def set_line_dash(self, line_dash_func: Callable) -> None:
        """
        Set that this styler should apply line dashes.

        :param Callable line_dash_func: a function returning the dashes
        """
        tmp = line_dash_func(self.count)
        if not isinstance(tmp, Iterable):
            raise TypeError(
                f"Line dashes must be iterable, but are {type(tmp)}.")
        self.__line_dashes = tuple(tmp)
        if len(self.__line_dashes) != self.count:
            raise ValueError(f"There must be {self.count} line dashes,"
                             f"but found only {len(self.__line_dashes)}.")
        self.has_style = True

    def set_line_width(self, line_width_func: Callable) -> None:
        """
        Set that this styler should apply a line width.

        :param Callable line_width_func: the line width function
        """
        tmp = line_width_func(self.count)
        if not isinstance(tmp, Iterable):
            raise TypeError(
                f"Line widths must be iterable, but are {type(tmp)}.")
        self.__line_widths = tuple(tmp)
        if len(self.__line_widths) != self.count:
            raise ValueError(f"There must be {self.count} line widths,"
                             f"but found only {len(self.__line_widths)}.")
        self.has_style = True

    def set_line_alpha(self, line_alpha_func: Callable) -> None:
        """
        Set that this styler should apply a line alpha.

        :param Callable line_alpha_func: the line alpha function
        """
        tmp = line_alpha_func(self.count)
        if not isinstance(tmp, Iterable):
            raise TypeError(
                f"Line alphas must be iterable, but are {type(tmp)}.")
        self.__line_alphas = tuple(tmp)
        if len(self.__line_alphas) != self.count:
            raise ValueError(f"There must be {self.count} line alphas,"
                             f"but found only {len(self.__line_alphas)}.")
        self.has_style = True

    def add_line_style(self, obj,
                       style: Dict[str, object]) -> None:
        """
        Apply this styler's contents based on the given object.

        :param obj: the object for which the style should be created
        :param Dict[str, object] style: the map to which the styles should be
            added
        """
        key = self.key_func(obj)
        index = self.__indexes.setdefault(key, -1)
        if index >= 0:
            self.__add_line_style(index, style)

    def __add_line_style(self, index,
                         style: Dict[str, object]) -> None:
        """
        Apply this styler's contents based on the given object.

        :param index: the index to be processed
        :param Dict[str, object] style: the map to which the styles should be
            added
        """
        if self.__line_colors is not None:
            style["color"] = self.__line_colors[index]
        if self.__line_dashes is not None:
            style["linestyle"] = self.__line_dashes[index]
        if self.__line_widths is not None:
            style["linewidth"] = self.__line_widths[index]
        if self.__line_alphas is not None:
            style["alpha"] = self.__line_alphas[index]

    def add_to_legend(self, collector: MutableSequence[Artist]) -> None:
        """
        Add this styler to the legend.

        :param MutableSequence[Tuple[Artist,Any]] collector: the collector to
            add to
        """
        for i, name in enumerate(self.names):
            style = create_line_style()
            self.__add_line_style(i, style)
            style["label"] = name
            style["xdata"] = []
            style["ydata"] = []
            collector.append(Line2D(**style))
