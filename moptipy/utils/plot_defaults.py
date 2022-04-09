"""Default styles for plots."""
from typing import Union, Tuple, cast, List, Final, Dict

import matplotlib.cm as mplcm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib import colors  # type: ignore

from moptipy.utils.types import type_error

#: The internal color black.
COLOR_BLACK: Final[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
#: The internal color white.
COLOR_WHITE: Final[Tuple[float, float, float]] = (1.0, 1.0, 1.0)


def default_name_func(key) -> str:
    """
    Compute the default name for a `key`: `str(key)`.

    :param key: the key
    :return: the key, too
    """
    return str(key)


#: A set of predefined uniquely-looking colors.
__FIXED_COLORS: Final[Tuple[Tuple[Tuple[float, float, float], ...], ...]] = \
    (((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.85, 0.0),
      (0.6392156862745098, 0.14901960784313725, 0.14901960784313725),
      (1.0, 0.6470588235294118, 0.0), (0.0, 0.5333333333333333, 0.2),
      (0.4666666666666667, 0.5333333333333333, 1.0),
      (0.8117647058823529, 0.0, 1.0), (0.4666666666666667, 0.0, 0.4),
      (0.8, 0.8, 0.0),
      (0.6274509803921569, 0.6627450980392157, 0.6274509803921569)),
     ((0.9019607843137255, 0.09803921568627451, 0.29411764705882354),
      (0.23529411764705882, 0.7058823529411765, 0.29411764705882354),
      (1.0, 0.8823529411764706, 0.09803921568627451),
      (0.0, 0.5098039215686274, 0.7843137254901961),
      (0.9607843137254902, 0.5098039215686274, 0.19215686274509805),
      (0.5686274509803921, 0.11764705882352941, 0.7058823529411765),
      (0.27450980392156865, 0.9411764705882353, 0.9411764705882353),
      (0.9411764705882353, 0.19607843137254902, 0.9019607843137255),
      (0.8235294117647058, 0.9607843137254902, 0.23529411764705882),
      (0.9803921568627451, 0.7450980392156863, 0.7450980392156863),
      (0.0, 0.5019607843137255, 0.5019607843137255),
      (0.9019607843137255, 0.7450980392156863, 1.0),
      (0.6666666666666666, 0.43137254901960786, 0.1568627450980392),
      (0.8117647058823529, 0.792156862745098, 0.03137254901960784),
      (0.5019607843137255, 0.0, 0.0),
      (0.6666666666666666, 1.0, 0.7647058823529411),
      (0.5019607843137255, 0.5019607843137255, 0.0),
      (1.0, 0.8470588235294118, 0.6941176470588235),
      (0.0, 0.0, 0.5019607843137255),
      (0.5176470588235295, 0.5176470588235295, 0.4627450980392157)),
     ((0.9019607843137255, 0.09803921568627451, 0.29411764705882354),
      (0.23529411764705882, 0.7058823529411765, 0.29411764705882354),
      (1.0, 0.8823529411764706, 0.09803921568627451),
      (0.0, 0.5098039215686274, 0.7843137254901961),
      (0.9607843137254902, 0.5098039215686274, 0.19215686274509805),
      (0.5686274509803921, 0.11764705882352941, 0.7058823529411765),
      (0.27450980392156865, 0.9411764705882353, 0.9411764705882353),
      (0.9411764705882353, 0.19607843137254902, 0.9019607843137255),
      (0.8235294117647058, 0.9607843137254902, 0.23529411764705882),
      (0.9803921568627451, 0.7450980392156863, 0.7450980392156863),
      (0.0, 0.5019607843137255, 0.5019607843137255),
      (0.9019607843137255, 0.7450980392156863, 1.0),
      (0.6666666666666666, 0.43137254901960786, 0.1568627450980392),
      (0.8117647058823529, 0.792156862745098, 0.03137254901960784),
      (0.5019607843137255, 0.0, 0.0),
      (0.6666666666666666, 1.0, 0.7647058823529411),
      (0.5019607843137255, 0.5019607843137255, 0.0),
      (1.0, 0.8470588235294118, 0.6941176470588235),
      (0.0, 0.0, 0.5019607843137255),
      (0.3137254901960784, 0.3137254901960784, 0.0),
      (0.5647058823529412, 0.6274509803921569, 0.6274509803921569)))


def distinct_colors(n: int) -> Tuple[Tuple[float, float, float], ...]:
    """
    Obtain a set of `n` distinct colors.

    :param n: the number of colors required
    :return: a tuple of colors
    """
    if not isinstance(n, int):
        raise type_error(n, "n", int)
    if not (0 < n < 1000):
        raise ValueError(f"Invalid n={n}.")

    # First, let us see if we can cover the range with hand-picked colors.
    for k in __FIXED_COLORS:
        lk = len(k)
        if lk >= n:
            if lk == n:
                return k
            return tuple(k[0:n])

    # Second, let's see whether the method from
    # https://stackoverflow.com/questions/8389636
    # works.
    cm = plt.get_cmap('gist_rainbow')
    c_norm = colors.Normalize(vmin=0, vmax=n - 1)
    scalar_map = mplcm.ScalarMappable(norm=c_norm, cmap=cm)
    qq = cast(List[Tuple[float, float, float]],
              [tuple(scalar_map.to_rgba(i)[0:3]) for i in range(n)])
    ss = set(qq)
    if len(ss) == n:
        return tuple(qq)

    raise ValueError(f"Could not obtain {n} distinct colors.")


#: The solid line dash
LINE_DASH_SOLID: Final[str] = "solid"

#: An internal array of fixed line styles.
__FIXED_LINE_DASHES: \
    Final[Tuple[Union[str, Tuple[float, Tuple[float, ...]]], ...]] = \
    tuple([LINE_DASH_SOLID,
           "dashed",
           "dashdot",
           "dotted",
           (0.0, (3.0, 5.0, 1.0, 5.0, 1.0, 5.0)),  # dashdotdotted
           (0.0, (3.0, 1.0, 1.0, 1.0)),  # densely dashdotted
           (0.0, (5.0, 1.0)),  # densely dashed
           (0.0, (1.0, 1.0)),  # densely dotted
           (0.0, (3.0, 1.0, 1.0, 1.0, 1.0, 1.0)),  # densely dashdotdotted
           (0.0, (1.0, 10.0)),  # loosely dotted
           (0.0, (5.0, 10.0)),  # loosely dashed
           (0.0, (3.0, 10.0, 1.0, 10.0)),  # loosely dashdotted
           (0.0, (3.0, 10.0, 1.0, 10.0, 1.0, 10.0))])  # loosely dashdotdotted


def distinct_line_dashes(n: int) -> \
        Tuple[Union[str, Tuple[float, Tuple[float, ...]]], ...]:
    """
    Create a sequence of distinct line dashes.

    :param n: the number of styles
    :return: the styles
    """
    if not isinstance(n, int):
        raise type_error(n, "n", int)
    if not (0 < n < len(__FIXED_LINE_DASHES)):
        raise ValueError(f"Invalid n={n} for line dash number, must be "
                         f"in 1..{len(__FIXED_LINE_DASHES)}.")
    if n == __FIXED_LINE_DASHES:
        return __FIXED_LINE_DASHES
    return tuple(__FIXED_LINE_DASHES[0:n])


#: The fixed predefined distinct markers
__FIXED_MARKERS: Tuple[str, ...] = ("o", "^", "s", "P", "X", "D", "*", "p")


def distinct_markers(n: int) -> Tuple[str, ...]:
    """
    Create a sequence of distinct markers.

    :param n: the number of markers
    :return: the markers
    """
    if not isinstance(n, int):
        raise type_error(n, "n", int)
    if not (0 < n < len(__FIXED_MARKERS)):
        raise ValueError(f"Invalid n={n} for line dash number, must be "
                         f"in 1..{len(__FIXED_MARKERS)}.")
    if n == __FIXED_MARKERS:
        return __FIXED_MARKERS
    return tuple(__FIXED_MARKERS[0:n])


def importance_to_line_width(importance: int) -> float:
    """
    Transform an importance value to a line width.

    Basically, an importance of `0` indicates a normal line in a normal
    plot that does not need to be emphasized.
    A positive importance means that the line should be emphasized.
    A negative importance means that the line should be de-emphasized.

    :param importance: a value between -9 and 9
    :return: the line width
    """
    if not isinstance(importance, int):
        raise type_error(importance, "importance", int)
    if not (-10 < importance < 10):
        raise ValueError(f"Invalid importance={importance}.")
    if importance >= 0:
        return 2.0 * (0.5 + importance)
    if importance == -1:
        return 2.0 / 3.0
    if importance == -2:
        return 0.5
    return 0.7 ** (-importance)


def importance_to_alpha(importance: int) -> float:
    """
    Transform an importance value to an alpha value.

    Basically, an importance of `0` indicates a normal line in a normal
    plot that does not need to be emphasized.
    A positive importance means that the line should be emphasized.
    A negative importance means that the line should be de-emphasized.

    :param importance: a value between -9 and 9
    :return: the alpha
    """
    if not isinstance(importance, int):
        raise type_error(importance, "importance", int)
    if not (-10 < importance < 10):
        raise ValueError(f"Invalid importance={importance}.")
    if importance >= 0:
        return 1.0
    if importance == -1:
        return 2.0 / 3.0
    if importance == -2:
        return 0.5
    return 1.0 / 3.0


#: The internal default basic style
__BASE_LINE_STYLE: Final[Dict[str, object]] = {
    "alpha": 1.0,
    "antialiased": True,
    "color": COLOR_BLACK,
    "dash_capstyle": "butt",
    "dash_joinstyle": "round",
    "linestyle": LINE_DASH_SOLID,
    "linewidth": 1.0,
    "solid_capstyle": "round",
    "solid_joinstyle": "round"
}


def create_line_style(**kwargs) -> Dict[str, object]:
    """
    Obtain the basic style for lines in diagrams.

    :param kwargs: any additional overrides
    :return: a dictionary with the style elements
    """
    res = dict(__BASE_LINE_STYLE)
    res.update(kwargs)
    return res


def importance_to_font_size(importance: float) -> float:
    """
    Transform an importance value to a font size.

    :param importance: the importance value
    :return: the font size
    """
    if not isinstance(importance, int):
        raise type_error(importance, "importance", int)
    if not (-10 < importance < 10):
        raise ValueError(f"Invalid importance={importance}.")
    if importance < 0:
        return 7.5
    if importance <= 0:
        return 8
    if importance == 1:
        return 8.5
    if importance == 2:
        return 9
    if importance == 3:
        return 10
    return 11


#: The default grid color
GRID_COLOR: Final[Tuple[float, float, float]] = \
    (7.0 / 11.0, 7.0 / 11.0, 7.0 / 11.0)


def rgb_to_gray(r: float, g: float, b: float) -> float:
    """
    Convert RGB values to gray scale.

    :param r: the red value
    :param g: the green value
    :param b: the blue value
    :return: the gray value
    """
    return (0.2989 * r) + (0.5870 * g) + (0.1140 * b)


def text_color_for_background(background: Tuple[float, float, float]) \
        -> Tuple[float, float, float]:
    """
    Get a reasonable text color for a given background color.

    :param background: the background color
    :return: the text color
    """
    br: Final[float] = background[0]
    bg: Final[float] = background[1]
    bb: Final[float] = background[2]
    bgg: Final[float] = rgb_to_gray(br, bg, bb)

    return COLOR_WHITE if bgg < 0.3 else COLOR_BLACK
