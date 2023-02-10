"""Default styles for plots."""
from typing import Final, cast

import matplotlib.cm as mplcm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib import colors  # type: ignore

from moptipy.utils.types import check_int_range, type_error

#: The internal color black.
COLOR_BLACK: Final[tuple[float, float, float]] = (0.0, 0.0, 0.0)
#: The internal color white.
COLOR_WHITE: Final[tuple[float, float, float]] = (1.0, 1.0, 1.0)


def str_to_palette(palette: str) \
        -> tuple[tuple[float, float, float], ...]:
    """
    Obtain a palette from a string.

    :param palette: the string with all the color data.
    :returns: the palette
    """
    if isinstance(colors, str):
        raise type_error(colors, "colors", str)
    result: list[tuple[float, float, float]] = []
    end: int = -1
    length: Final[int] = len(palette)
    while end < length:
        start: int = end + 1
        end = length
        for ch in "\n\r\t ,;":
            end_new: int = palette.find(ch, start)
            if start < end_new < end:
                end = end_new
        color: str = palette[start:end].strip()
        if color.startswith("#"):
            color = color[1:].strip()
        if len(color) > 0:
            if len(color) != 6:
                raise ValueError(f"invalid color: {color!r} in {palette!r}")
            result.append((int(color[0:2], 16) / 255,
                           int(color[2:4], 16) / 255,
                           int(color[4:6], 16) / 255))
    if len(result) <= 0:
        raise ValueError(f"empty colors: {palette!r}")
    return tuple(result)


#: A palette with 11 distinct colors.
__PALETTE_11: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "ff0000 0000ff 00d900 a32626 ffa500 008833 7788ff cf00ff 770066 cccc00 "
    "a0a9a0")

#: A palette with 21 distinct colors.
__PALETTE_21: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "e6194b 3cb44b ffe119 0082c8 f58231 911eb4 46f0f0 f032e6 d2f53c fabebe "
    "008080 e6beff aa6e28 cfca08 800000 aaffc3 808000 ffd8b1 000080 505000 "
    "90a0a0")

#: A palette with 25 distinct colors.
__PALETTE_25: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "20b2aa f0e68c b8860b ff00ff 00fa9a ee82ee 9acd32 dcdcdc 556b2f db7093 "
    "00bfff 1e90ff 8a2be2 ff1493 008000 ffd700 000080 800000 7fff00 8b008b "
    "0000ff 483d8b ff0000 696969 ff7f50")

#: A palette with 28 distinct colors.
__PALETTE_28: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "545454 550000 b0b0ff bababa 878787 005500 550055 b000b0 e4e400 00b0b0 "
    "00ffff 00ff00 878700 870087 005555 ff0000 0000ff baba00 00b000 ff00ff "
    "870000 4949ff 008700 8484ff 008787 e4e4e4 545400 b00000")

#: A palette with 30 distinct colors.
__PALETTE_30: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "00bfff ff1493 7f007f 8a2be2 1e90ff adff2f ffb6c1 fa8072 ffff54 90ee90 "
    "ff4500 00ff00 483d8b ffa500 696969 add8e6 ff00ff 7f0000 8fbc8f da70d6 "
    "006400 dc143c 00ff7f 008b8b 808000 b03060 00ffff 0000ff 000080 deb887 ")

#: A palette with 35 distinct colors.
__PALETTE_35: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "228b22 808080 9acd32 008080 483d8b 7f0000 d2b48c 00ff00 00008b 808000 "
    "00ced1 8fbc8f 556b2f ffc0cb 1e90ff ffff54 800080 00bfff ff00ff 7b68ee "
    "ff6347 9400d3 ffa500 7fffd4 b03060 ff0000 f08080 f4a460 00ff7f 98fb98 "
    "f0e68c ff1493 0000ff b0e0e6 dda0dd")

#: A palette with 40 distinct colors.
__PALETTE_40: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "ff00ff 7f007f a9a9a9 0000ff daa520 2f4f4f da70d6 ff8c00 ffc0cb afeeee "
    "00008b 20b2aa 00ff00 f4a460 8b0000 008000 f08080 a0522d ff6347 808000 "
    "bdb76b 98fb98 191970 ffe4c4 ffff00 4682b4 32cd32 1e90ff 3cb371 ff1493 "
    "b03060 7fffd4 556b2f dda0dd adff2f 7b68ee 9acd32 ff0000 00bfff 9400d3 ")

#: A palette with 45 distinct colors.
__PALETTE_45: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "ffb60c 00008b 8b4513 e9967a a020f0 ff0000 6a5acd 808080 008b8b 00ff00 "
    "90ee90 4682b4 daa520 7f007f ffd700 00bfff 00ced1 556b2f c0c0c0 008000 "
    "ff69b4 dda0dd adff2f 3cb371 8b0000 0000ff f5deb3 32cd32 1e90ff b03060 "
    "ff7f50 cd5c5c ff00ff 7fffd4 6b8e23 dc143c 9acd32 afeeee 2f4f4f 00fa9a "
    "191970 8fbc8f ba55d3 ff8c00 bdb76b")

#: A palette with 50 distinct colors.
__PALETTE_50: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "7fffd4 000080 00ff7f 483d8b 008000 afeeee ffc0cb daa520 ffd700 ff8c00 "
    "87cefa ff00ff c0c0c0 cd5c5c 00ff00 adff2f e9967a ffff00 dc143c cd853f "
    "9932cc 98fb98 708090 800000 ee82ee 800080 8fbc8f 4682b4 dda0dd 32cd32 "
    "a020f0 48d1cc a0522d ff1493 ff69b4 f0e68c 9370db 2f4f4f ff0000 6b8e23 "
    "556b2f ff6347 f5deb3 008b8b db7093 0000cd 9acd32 3cb371 0000ff 6495ed")

#: A palette with 60 distinct colors.
__PALETTE_60: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "d2b48c 191970 000080 20b2aa bc8f8f 2f4f4f 8b4513 ff4500 dc143c afeeee "
    "db7093 c71585 bdb76b b03060 708090 00ffff 6495ed 808000 f08080 b8860b "
    "cd853f 556b2f ffa07a 008080 eee8aa dda0dd ee82ee 7fffd4 9370db 7f007f "
    "d8bfd8 9acd32 4682b4 98fb98 00ff00 9932cc 0000cd d2691e 00fa9a ff8c00 "
    "663399 a9a9a9 ffff00 32cd32 8b0000 4169e1 8fbc8f 0000ff ff69b4 008000 "
    "ffb6c1 00bfff ffd700 ff1493 ff00ff 3cb371 a020f0 87ceeb ff6347 adff2f")

#: A palette with 70 distinct colors.
__PALETTE_70: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "b8860b a52a2a ffe4c4 663399 da70d6 a020f0 eee8aa 9acd32 ba55d3 ff8c00 "
    "00008b 2e8b57 66cdaa 2f4f4f 0000cd 32cd32 556b2f adff2f 7fffd4 00ff00 "
    "008b8b 4b0082 ffd700 00bfff 6b8e23 48d1cc b03060 778899 ffff54 ff0000 "
    "00ff7f d2b48c 3cb371 bc8f8f b0e0e6 ffc0cb dcdcdc ff00ff cd5c5c 6495ed "
    "0000ff 9370db dc143c a9a9a9 bdb76b f4a460 c71585 e9967a 4682b4 db7093 "
    "483d8b 8b008b 87cefa 7fff00 dda0dd 006400 9932cc 808000 ff1493 191970 "
    "4169e1 ff6347 b0c4de ff69b4 d2691e 00ffff 98fb98 ffff00 a0522d 8fbc8f")

#: A palette with 80 distinct colors.
__PALETTE_80: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "eee8aa bdb76b 2e8b57 0000ff 1e90ff cd5c5c ffd700 90ee90 00ffff ba55d3 "
    "d3d3d3 40e0d0 ff0000 b0c4de 008000 0000cd 00bfff ffa07a ff00ff 778899 "
    "dc143c cd853f 556b2f 9acd32 9400d3 663399 b03060 ffdab9 b22222 ffa500 "
    "7fffd4 4169e1 dda0dd 8fbc8f ffff00 ff1493 191970 00008b 9932cc 006400 "
    "483d8b 808000 696969 00ff7f d8bfd8 d2691e 4682b4 a9a9a9 c71585 5f9ea0 "
    "ffc0cb 7f0000 800080 ee82ee 008080 4b0082 ffff54 9370db 8b4513 ff6347 "
    "daa520 87cefa adff2f ff8c00 6b8e23 deb887 00fa9a 66cdaa 7b68ee fa8072 "
    "ff69b4 db7093 b0e0e6 00ff00 2f4f4f bc8f8f 32cd32 3cb371 20b2aa 7cfc00")

#: A palette with 90 distinct colors.
__PALETTE_90: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "fa8072 ffff54 0000cd 32cd32 dc143c 40e0d0 ff8c00 9932cc 7b68ee 800000 "
    "4b0082 008000 afeeee ee82ee ffd700 6a5acd 2e8b57 eee8aa a9a9a9 ff69b4 "
    "b8860b c71585 ff4500 7fff00 2f4f4f 00ff7f ffa500 add8e6 008b8b bdb76b "
    "ff00ff cd853f f0e68c 808080 db7093 800080 a0522d 87ceeb 3cb371 9400d3 "
    "b0c4de daa520 b03060 dda0dd 66cdaa 6b8e23 d3d3d3 4682b4 9acd32 4169e1 "
    "ff7f50 5f9ea0 f08080 00fa9a 00ced1 808000 adff2f 0000ff d2691e ff1493 "
    "ff6347 ffdead 20b2aa 006400 7fffd4 483d8b cd5c5c f4a460 8fbc8f 663399 "
    "d8bfd8 9370db b22222 d2b48c ffff00 ba55d3 bc8f8f ffb6c1 6495ed e9967a "
    "ff0000 000080 00ff00 191970 90ee90 00ffff 778899 1e90ff 00bfff 556b2f")

#: A palette with 100 distinct colors.
__PALETTE_100: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "bdb76b 8a2be2 bc8f8f eee8aa 8fbc8f a020f0 ff4500 2f4f4f 000080 ff6347 "
    "6b8e23 808080 663399 afeeee dcdcdc ff8c00 3cb371 191970 4682b4 7cfc00 "
    "ee82ee 32cd32 9370db ff1493 00ff00 6a5acd adff2f 006400 00fa9a ffd700 "
    "d2691e c0c0c0 00ff7f ba55d3 add8e6 ffff00 6495ed 8b008b 48d1cc 87cefa "
    "fa8072 deb887 b22222 7b68ee b8860b 90ee90 8b4513 9acd32 cd853f f08080 "
    "ff00ff da70d6 9400d3 ff69b4 ffe4b5 556b2f 8b0000 1e90ff e9967a 696969 "
    "c71585 ffa07a f0e68c a9a9a9 20b2aa a0522d 87ceeb 0000ff 008000 daa520 "
    "00ffff b03060 483d8b ffb6c1 dda0dd ff0000 9932cc 708090 cd5c5c 4b0082 "
    "ffa500 0000cd dc143c a52a2a 008080 7fffd4 5f9ea0 b0c4de d8bfd8 808000 "
    "db7093 f4a460 2e8b57 ffff54 40e0d0 66cdaa ffe4c4 ff7f50 4169e1 00bfff")

#: A palette with 110 distinct colors.
__PALETTE_110: Final[tuple[tuple[float, float, float], ...]] = str_to_palette(
    "0000ff 556b2f ff4500 000080 f5deb3 ff1493 00008b 0000cd ff8c00 ffc0cb "
    "fa8072 696969 da70d6 8b4513 d2691e 00fa9a 008080 c71585 bdb76b 40e0d0 "
    "6495ed daa520 5f9ea0 ffe4c4 f4a460 ba55d3 228b22 00ced1 00ffff 66cdaa "
    "b03060 9400d3 f08080 dc143c adff2f 00ff00 9932cc ee82ee 8a2be2 32cd32 "
    "afeeee 191970 808080 9acd32 90ee90 b22222 a0522d ff00ff 006400 c0c0c0 "
    "2f4f4f d3d3d3 cd853f 98fb98 9370db 4169e1 4b0082 00ff7f ff7f50 8fbc8f "
    "8b0000 eee8aa dda0dd 008000 ffa500 ff6347 ff69b4 d8bfd8 ffa07a 4682b4 "
    "b0c4de 2e8b57 ff0000 00bfff f0e68c 7f0000 b8860b d2b48c ffff54 1e90ff "
    "e9967a 3cb371 87cefa 87ceeb 6a5acd ffd700 708090 bc8f8f 7b68ee 6b8e23 "
    "a52a2a 7fffd4 ffdead ffdab9 808000 483d8b 20b2aa a020f0 48d1cc db7093 "
    "a9a9a9 663399 b0e0e6 deb887 7f007f 7fff00 8b008b ffff00 cd5c5c add8e6")


#: A set of predefined uniquely-looking colors.
__FIXED_COLORS: Final[tuple[tuple[tuple[float, float, float], ...], ...]] = (
    __PALETTE_11, __PALETTE_21, __PALETTE_25, __PALETTE_28, __PALETTE_30,
    __PALETTE_35, __PALETTE_40, __PALETTE_45, __PALETTE_50, __PALETTE_60,
    __PALETTE_70, __PALETTE_80, __PALETTE_90, __PALETTE_100, __PALETTE_110)


def distinct_colors(n: int) -> tuple[tuple[float, float, float], ...]:
    """
    Obtain a set of `n` distinct colors.

    :param n: the number of colors required
    :return: a tuple of colors
    """
    check_int_range(n, "n", 1, 1000)

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
    # This method does not seem to make good use of the available color space.
    # Since we use it only for cases with more than 110 colors, that's OK.
    cm = plt.get_cmap("gist_rainbow")
    c_norm = colors.Normalize(vmin=0, vmax=n - 1)
    scalar_map = mplcm.ScalarMappable(norm=c_norm, cmap=cm)
    qq = cast(list[tuple[float, float, float]],
              [tuple(scalar_map.to_rgba(i)[0:3]) for i in range(n)])
    ss = set(qq)
    if len(ss) == n:
        return tuple(qq)

    raise ValueError(f"Could not obtain {n} distinct colors.")


#: The solid line dash
LINE_DASH_SOLID: Final[str] = "solid"

#: An internal array of fixed line styles.
__FIXED_LINE_DASHES: \
    Final[tuple[str | tuple[float, tuple[float, ...]], ...]] = \
    (LINE_DASH_SOLID,
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
     (0.0, (3.0, 10.0, 1.0, 10.0, 1.0, 10.0)))  # loosely dashdotdotted


def distinct_line_dashes(n: int) -> \
        tuple[str | tuple[float, tuple[float, ...]], ...]:
    """
    Create a sequence of distinct line dashes.

    :param n: the number of styles
    :return: the styles
    """
    check_int_range(n, "n", 1, len(__FIXED_LINE_DASHES) - 1)
    if n == __FIXED_LINE_DASHES:
        return __FIXED_LINE_DASHES
    return tuple(__FIXED_LINE_DASHES[0:n])


#: The fixed predefined distinct markers
__FIXED_MARKERS: tuple[str, ...] = ("o", "^", "s", "P", "X", "D", "*", "p")


def distinct_markers(n: int) -> tuple[str, ...]:
    """
    Create a sequence of distinct markers.

    :param n: the number of markers
    :return: the markers
    """
    lfm: Final[int] = len(__FIXED_MARKERS)
    check_int_range(n, "n", 1, lfm)
    if n == lfm:
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
    check_int_range(importance, "importance", -9, 9)
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
    check_int_range(importance, "importance", -9, 9)
    if importance >= 0:
        return 1.0
    if importance == -1:
        return 2.0 / 3.0
    if importance == -2:
        return 0.5
    return 1.0 / 3.0


#: The internal default basic style
__BASE_LINE_STYLE: Final[dict[str, object]] = {
    "alpha": 1.0,
    "antialiased": True,
    "color": COLOR_BLACK,
    "dash_capstyle": "butt",
    "dash_joinstyle": "round",
    "linestyle": LINE_DASH_SOLID,
    "linewidth": 1.0,
    "solid_capstyle": "round",
    "solid_joinstyle": "round",
}


def create_line_style(**kwargs) -> dict[str, object]:
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
    check_int_range(importance, "importance", -9, 9)
    if importance < 0:
        return 7.5
    if importance <= 0:
        return 8.0
    if importance == 1:
        return 8.5
    if importance == 2:
        return 9.0
    if importance == 3:
        return 10.0
    return 11.0


#: The default grid color
GRID_COLOR: Final[tuple[float, float, float]] = \
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


def text_color_for_background(background: tuple[float, float, float]) \
        -> tuple[float, float, float]:
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
