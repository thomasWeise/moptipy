"""Utilities for creating and storing figures."""
import os.path
from math import sqrt, isfinite
from typing import Final, Iterable, Union, List, Optional, Callable, cast, \
    Tuple, Sequence

import matplotlib.pyplot  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.backend_bases import RendererBase  # type: ignore
from matplotlib.backends.backend_agg import RendererAgg  # type: ignore
from matplotlib.figure import Figure, SubplotBase  # type: ignore

import moptipy.evaluation.plot_defaults as pd
from moptipy.evaluation.lang import Lang
from moptipy.utils.path import Path

#: the golden ratio
__GOLDEN_RATIO: Final[float] = 0.5 + (0.5 * sqrt(5))


def create_figure(width: Union[float, int, None] = 8.6,
                  height: Union[float, int, None] = None,
                  dpi: Union[float, int, None] = 384.0,
                  **kwargs) -> Figure:
    """
    Create a matplotlib figure.

    :param Union[float,int,None] width: the optional width
    :param Union[float,int,None] height: the optional height
    :param Union[float,int,None] dpi: the dpi value
    :param kwargs: a set of optional arguments
    :return: the figure option
    :rtype: Figure
    """
    # Check the size, i.e., width and height and use
    put_size: bool = True
    if width is None:
        if height is None:
            put_size = False
        else:
            width = __GOLDEN_RATIO * height
    elif height is None:
        height = width / __GOLDEN_RATIO

    if put_size:
        if isinstance(height, int):
            height = float(height)
        if not isinstance(height, float):
            raise ValueError(f"Invalid height type {type(height)}.")
        if not (isfinite(height) and (0.1 < height < 10000.0)):
            raise ValueError(f"Invalid height {height}.")
        nheight = int(0.5 + (height * 72)) / 72.0
        if not (isfinite(nheight) and (0.1 < nheight < 10000.0)):
            raise ValueError(f"Invalid height {height} as it maps to "
                             f"{nheight} after to-point-and-back conversion.")

        if isinstance(width, int):
            width = float(width)
        if not isinstance(width, float):
            raise ValueError(f"Invalid width type {type(width)}.")
        if not (isfinite(width) and (0.1 < width < 10000.0)):
            raise ValueError(f"Invalid width {width}.")
        nwidth = int(0.5 + (width * 72)) / 72.0
        if not (isfinite(nwidth) and (0.1 < nwidth < 10000.0)):
            raise ValueError(f"Invalid width {width} as it maps to "
                             f"{nwidth} after to-point-and-back conversion.")

        kwargs['figsize'] = width, height

    if dpi is not None:
        if isinstance(dpi, int):
            dpi = float(dpi)
        if not isinstance(dpi, float):
            raise ValueError(f"Invalid dpi type {type(dpi)}.")
        if not (isfinite(dpi) and (1.0 < dpi < 10000.0)):
            raise ValueError(f"Invalid dpi value {dpi}.")
        kwargs['dpi'] = dpi

    if 'frameon' not in kwargs:
        kwargs['frameon'] = False

    if 'constrained_layout' not in kwargs:
        kwargs['constrained_layout'] = False

    return Figure(**kwargs)


def save_figure(fig: Figure,
                file_name: str = "figure",
                dir_name: str = ".",
                formats: Union[str, Iterable[str]] = "svg") -> List[Path]:
    """
    Store the given figure in files of the given formats and dispose it.

    :param Figure fig: the figure to save
    :param str file_name: the basic file name
    :param str dir_name: the directory name
    :param Union[str, Iterable[str]] formats: the file format(s)
    :return: a list of files
    :rtype: List[str]
    """
    if not isinstance(fig, Figure):
        raise TypeError(f"Invalid figure type {type(fig)}.")
    if not isinstance(file_name, str):
        raise TypeError(f"Invalid file_name type {type(file_name)}.")
    if len(file_name) <= 0:
        raise ValueError(f"Invalid filename '{file_name}'.")
    if not isinstance(dir_name, str):
        raise TypeError(f"Invalid dirname type {type(dir_name)}.")
    if len(dir_name) <= 0:
        raise ValueError(f"Invalid dirname '{dir_name}'.")
    if isinstance(formats, str):
        formats = [formats]
    if not isinstance(formats, Iterable):
        raise TypeError(f"Invalid format type {type(formats)}.")

    size = fig.get_size_inches()
    if size[0] >= size[1]:
        orientation = "landscape"
    else:
        orientation = "portrait"

    # set minimal margins to the axes to avoid wasting space
    for ax in fig.axes:
        ax.margins(0, 0)

    use_dir = Path.directory(dir_name)
    files = []
    for fmt in formats:
        if not isinstance(fmt, str):
            raise TypeError(f"Invalid format type {type(fmt)}.")
        dest_file = Path.path(os.path.join(use_dir, f"{file_name}.{fmt}"))
        dest_file.ensure_file_exists()
        fig.savefig(dest_file, transparent=True, format=fmt,
                    orientation=orientation,
                    dpi="figure",
                    bbox_inches='tight',
                    pad_inches=1.0 / 72.0)
        dest_file.enforce_file()
        files.append(dest_file)

    fig.clf(False)
    matplotlib.pyplot.close(fig)
    del fig

    if len(files) <= 0:
        raise ValueError("No formats were specified.")

    return files


def label_box(axes: Axes,
              text: str,
              x: Optional[float] = None,
              y: Optional[float] = None,
              font_size: float = pd.importance_to_font_size(0),
              may_rotate_text: bool = False,
              zorder: Optional[float] = None,
              font: Union[None, str, Callable] =
              lambda: Lang.current().font()) -> None:
    """
    Put a label text box near an axis.

    :param Axes axes: the axes to add the label to
    :param str text: the text to place
    :param Optional[float] x: the location along the x-axis: `0` means left,
        `0.5` means centered, `1` means right
    :param Optional[float] y: the location along the x-axis: `0` means bottom,
        `0.5` means centered, `1` means top
    :param float font_size: the font size
    :param bool may_rotate_text: should we rotate the text by 90° if that
        makes sense (`True`) or always keep it horizontally (`False`)
    :param Optional[float] zorder: an optional z-order value
    :param Union[None, str, Callable] font: the font to use
    """
    if x is None:
        if y is None:
            raise ValueError("At least one of x or y must not be None.")
        x = 0
    elif y is None:
        y = 0

    spacing: Final[float] = max(4.0, font_size / 2.0)
    xtext: float = 0.0
    ytext: float = 0.0
    xalign: str = "center"
    yalign: str = "center"
    if x >= 0.85:
        xtext = -spacing
        xalign = "right"
    elif x <= 0.15:
        xtext = spacing
        xalign = "left"
    if y >= 0.85:
        ytext = -spacing
        yalign = "top"
    elif y <= 0.15:
        ytext = spacing
        yalign = "bottom"

    args = {"text": text,
            "xy": (x, y),
            "xytext": (xtext, ytext),
            "verticalalignment": yalign,
            "horizontalalignment": xalign,
            "xycoords": "axes fraction",
            "textcoords": "offset points",
            "fontsize": font_size,
            "bbox": {"boxstyle": 'round',
                     "color": 'white',
                     "fill": True,
                     "linewidth": 0,
                     "alpha": 0.9}}
    if zorder is not None:
        args["zorder"] = zorder

    if may_rotate_text and (len(text) > 2):
        args["rotation"] = 90

    if callable(font):
        font = font()
    if font is not None:
        if not isinstance(font, str):
            raise TypeError(f"Font must be string, but is {type(font)}.")
        args['fontname'] = font

    axes.annotate(**args)


def label_axes(axes: Axes,
               xlabel: Optional[str] = None,
               xlabel_inside: bool = True,
               xlabel_location: float = 0.5,
               ylabel: Optional[str] = None,
               ylabel_inside: bool = True,
               ylabel_location: float = 1,
               font_size: float = pd.importance_to_font_size(0),
               zorder: Optional[float] = None) -> None:
    """
    Put labels on a figure.

    :param Axes axes: the axes to add the label to
    :param Optional[str] xlabel: a callable returning the label for
        the x-axis, a label string, or `None` if no label should be put
    :param bool xlabel_inside: put the x-axis label inside the plot (so that
        it does not consume additional vertical space)
    :param float xlabel_location: the location of the x-axis label if it is
        placed inside the plot area
    :param Optional[str] ylabel: a callable returning the label for
        the y-axis, a label string, or `None` if no label should be put
    :param bool ylabel_inside: put the xyaxis label inside the plot (so that
        it does not consume additional horizontal space)nal vertical space)
    :param float ylabel_location: the location of the y-axis label if it is
        placed inside the plot area
    :param float font_size: the font size to use
    :param Optional[float] zorder: an optional z-order value
    """
    # put the label on the x-axis, if any
    if xlabel is not None:
        if not isinstance(xlabel, str):
            raise TypeError(f"xlabel must be str but is {type(xlabel)}.")
        if len(xlabel) > 0:
            if xlabel_inside:
                label_box(axes, text=xlabel, x=xlabel_location, y=0,
                          font_size=font_size, zorder=zorder)
            else:
                axes.set_xlabel(xlabel, fontsize=font_size)

    # put the label on the y-axis, if any
    if ylabel is not None:
        if not isinstance(ylabel, str):
            raise TypeError(f"ylabel must be str but is {type(ylabel)}.")
        if len(ylabel) > 0:
            if ylabel_inside:
                label_box(axes, text=ylabel, x=0, y=ylabel_location,
                          font_size=font_size, may_rotate_text=True,
                          zorder=zorder)
            else:
                axes.set_ylabel(ylabel, fontsize=font_size)


def divide_into_sub_plots(figure: Figure, nrows: int, ncols: int) \
        -> Tuple[Tuple[Union[SubplotBase, Figure], int, int, int], ...]:
    """
    Divide a figure into nrows*ncols sub-plots.

    :param Figure figure: the figure
    :param int nrows: the number of rows
    :param int ncols: the number of columns
    :returns: a tuple with the sub-figures, their row, their column,
        and their overall index
    :rtype: Tuple[Tuple[Union[SubplotBase, Figure], int, int, int], ...]
    """
    if not isinstance(figure, Figure):
        raise TypeError(f"Expected Figure, but got {type(figure)}.")
    if not isinstance(nrows, int):
        raise TypeError(f"nrows must be int, but is {type(nrows)}.")
    if nrows < 1:
        raise ValueError(f"nrows must be positive, but is {nrows}.")
    if not isinstance(ncols, int):
        raise TypeError(f"ncols must be int, but is {type(ncols)}.")
    if ncols < 1:
        raise ValueError(f"ncols must be positive, but is {ncols}.")

    if (nrows == 1) and (ncols == 1):
        return tuple([(figure, 0, 0, 0)])

    allfigs: List[Tuple[Union[SubplotBase, Figure], int, int, int]] = []
    index: int = 0
    for i in range(nrows):
        for j in range(ncols):
            index += 1
            allfigs.append((figure.add_subplot(nrows, ncols, index),
                            i, j, index - 1))
    return tuple(allfigs)


def get_axes(figure: Union[Axes, SubplotBase, Figure]) -> Axes:
    """
    Obtain the axes from a figure or axes object.

    :param Union[SubplotBase, Figure] figure: the figure
    :return: the Axes
    :rtype: Axes
    """
    if isinstance(figure, Figure):
        return figure.add_axes([0.005, 0.005, 0.99, 0.99])
    if hasattr(figure, 'axes') \
            or isinstance(getattr(type(figure), 'axes', None), property):
        try:
            if isinstance(figure.axes, Axes):
                return cast(Axes, figure.axes)
            if isinstance(figure.axes, Sequence):
                ax = figure.axes[0]
                if isinstance(ax, Axes):
                    return cast(Axes, ax)
            elif isinstance(figure.axes, Iterable):
                for k in figure.axes:
                    if isinstance(k, Axes):
                        return cast(Axes, k)
                    break
        except TypeError:
            pass
        except IndexError:
            pass
    if isinstance(figure, Axes):
        return cast(Axes, figure)
    raise TypeError(f"Cannot get Axes of object of type {type(figure)}.")


def get_renderer(figure: Union[SubplotBase, Axes, Figure]) -> RendererBase:
    """
    Get a renderer that can be used for determining figure element sizes.

    :param Union[SubplotBase, Figure] figure: the figure element
    :return: the renderer
    :rtype: RendererBase
    """
    if isinstance(figure, (Axes, SubplotBase)):
        figure = figure.figure
    if not isinstance(figure, Figure):
        raise TypeError(f"Figure expected, but got {type(figure)}.")
    canvas = figure.canvas
    if hasattr(canvas, "renderer"):
        return canvas.renderer
    if hasattr(canvas, "get_renderer"):
        return canvas.get_renderer()
    return RendererAgg(width=figure.get_figwidth(),
                       height=figure.get_figheight(),
                       dpi=figure.get_dpi())


def cm_to_inch(cm: Union[int, float]) -> float:
    """
    Convert cm to inch.

    :param float cm: the cm value
    :return: the value in inch
    :rtype: float
    """
    if not isinstance(cm, int):
        if not isinstance(cm, float):
            raise TypeError(f"cm must be int or float, but is {type(cm)}.")
        if not isfinite(cm):
            raise ValueError(f"cm must be finite, but is {cm}.")
    res: float = cm / 2.54
    if not isfinite(res):
        raise ValueError(f"Conversation {cm} cm to inch "
                         f"must be finite, but is {res}.")
    return res
