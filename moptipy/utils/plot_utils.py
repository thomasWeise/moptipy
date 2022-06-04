"""Utilities for creating and storing figures."""
import os.path
import statistics as st
from math import sqrt, isfinite
from typing import Final, Iterable, Union, List, Optional, Callable, cast, \
    Tuple, Sequence, Dict

import matplotlib.pyplot  # type: ignore
from matplotlib.artist import Artist  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.backend_bases import RendererBase  # type: ignore
from matplotlib.backends.backend_agg import RendererAgg  # type: ignore
from matplotlib.figure import Figure, SubplotBase  # type: ignore

import moptipy.utils.plot_defaults as pd
from moptipy.utils.lang import Lang
from moptipy.utils.path import Path
from moptipy.utils.types import type_error, type_name_of

#: the golden ratio
__GOLDEN_RATIO: Final[float] = 0.5 + (0.5 * sqrt(5))


def create_figure(width: Union[float, int, None] = 8.6,
                  height: Union[float, int, None] = None,
                  dpi: Union[float, int, None] = 384.0,
                  **kwargs) -> Figure:
    """
    Create a matplotlib figure.

    :param width: the optional width
    :param height: the optional height
    :param dpi: the dpi value
    :param kwargs: a set of optional arguments
    :return: the figure option
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


def __divide_evenly(items: int, chunks: int, reverse: bool) -> List[int]:
    """
    Divide `n` items into `k` chunks, trying to create equally-sized chunks.

    :param items: the number of items to divide
    :param chunks: the number of chunks
    :param reverse: should we put the additional items at the end (True)
        or front (False)
    :returns: the list of items

    >>> print(__divide_evenly(9, 3, reverse=True))
    [3, 3, 3]
    >>> print(__divide_evenly(10, 3, reverse=True))
    [3, 3, 4]
    >>> print(__divide_evenly(10, 3, reverse=False))
    [4, 3, 3]
    >>> print(__divide_evenly(11, 3, reverse=True))
    [3, 4, 4]
    >>> print(__divide_evenly(11, 3, reverse=False))
    [4, 4, 3]
    >>> print(__divide_evenly(12, 3, reverse=False))
    [4, 4, 4]
    """
    # validate that we do not have more chunks than items or otherwise invalid
    # parameters
    if (items <= 0) or (chunks <= 0) or (chunks > items):
        raise ValueError(f"cannot divide {items} items into {chunks} chunks.")

    # First we compute the minimum number of items per chunk.
    # Our basic solution is to put exactly this many items into each chunk.
    # For example, if items=10 and chunks=3, this will fill 3 items into each
    # chunk.
    result: List[int] = [items // chunks] * chunks
    # We then fill the remaining items into the chunks at the front
    # (reverse=False) or end (reverse=True) of the list.
    # We do this by putting exactly one such item into each chunk, starting
    # from the end.
    # Since items modulo chunks must be in [0..chunks), this is possible.
    # This setup will yield the lowest possible standard deviation of items
    # per chunk.
    # In case of items=10 and chunks=3, items % chunks = 1 and the chunks will
    # become [3, 3, 4]. If items=3, no items need to be distributed and we get
    # [3, 3, 3].
    if reverse:
        for i in range(1, (items % chunks) + 1):
            result[-i] += 1
    else:
        for i in range(items % chunks):
            result[i] += 1
    return result


def create_figure_with_subplots(
        items: int,
        max_items_per_plot: int = 3,
        max_rows: int = 3,
        max_cols: int = 1,
        min_rows: int = 1,
        min_cols: int = 1,
        default_width_per_col: Union[float, int, None] = 8.6,
        max_width: Union[float, int, None] = 8.6,
        default_height_per_row: Union[float, int, None] = 8.6 / __GOLDEN_RATIO,
        max_height: Union[float, int, None] = 9,
        dpi: Union[float, int, None] = 384.0,
        **kwargs) \
        -> Tuple[Figure, Tuple[Tuple[Union[SubplotBase, Figure],
                                     int, int, int, int, int], ...]]:
    """
    Divide a figure into nrows*ncols sub-plots.

    :param items: the number of items to divide
    :param max_items_per_plot: the maximum number of items per plot
    :param max_rows: the maximum number of rows
    :param max_cols: the maximum number of columns
    :param min_rows: the minimum number of rows
    :param min_cols: the minimum number of cols
    :param default_width_per_col: the optional default width of a column
    :param default_height_per_row: the optional default height per row
    :param max_height: the maximum height
    :param max_width: the maximum width
    :param dpi: the dpi value
    :param kwargs: a set of optional arguments
    :returns: a tuple with the figure, followed by a series of tuples with
        each sub-figure, the index of the first item assigned to it, the
        index of the last item assigned to it + 1, their row, their column,
        and their overall index
    """
    # First, we do a lot of sanity checks
    if not isinstance(items, int):
        raise type_error(items, "items", int)
    if items < 1:
        raise ValueError(f"items must be positive, but is {items}.")
    if not isinstance(max_items_per_plot, int):
        raise type_error(max_items_per_plot, "max_items_per_plot", int)
    if max_items_per_plot < 1:
        raise ValueError(f"max_items_per_plot must be positive, "
                         f"but is {max_items_per_plot}.")
    if not isinstance(max_rows, int):
        raise type_error(max_rows, "max_rows", int)
    if max_rows < 1:
        raise ValueError(f"max_rows must be positive, but is {max_rows}.")
    if not isinstance(min_rows, int):
        raise type_error(min_rows, "min_rows", int)
    if min_rows < 1:
        raise ValueError(f"min_rows must be positive, but is {min_rows}.")
    if min_rows > max_rows:
        raise ValueError(
            f"min_rows ({min_rows}) must be <= max_rows ({max_rows}).")
    if not isinstance(max_cols, int):
        raise type_error(max_cols, "max_cols", int)
    if max_cols < 1:
        raise ValueError(f"max_cols must be positive, but is {max_cols}.")
    if not isinstance(min_cols, int):
        raise type_error(min_cols, "min_cols", int)
    if min_cols < 1:
        raise ValueError(f"min_cols must be positive, but is {min_cols}.")
    if min_cols > max_cols:
        raise ValueError(
            f"min_cols ({min_cols}) must be <= max_cols ({max_cols}).")
    if (max_cols * max_rows * max_items_per_plot) < items:
        raise ValueError(
            f"Cannot distribute {items} items into at most {max_rows} rows "
            f"and {max_cols} cols with at most {max_items_per_plot} per "
            "plot.")
    if default_width_per_col is not None:
        default_width_per_col = float(default_width_per_col)
        if (not isfinite(default_width_per_col)) \
                or (default_width_per_col <= 0.1) \
                or (default_width_per_col >= 1000):
            raise ValueError(
                f"invalid default_width_per_col {default_width_per_col}")
    if max_width is not None:
        max_width = float(max_width)
        if (not isfinite(max_width)) \
                or (max_width <= 0.1) or (max_width >= 10000):
            raise ValueError(f"invalid max_width {max_width}")
        if (default_width_per_col is not None) \
                and (default_width_per_col > max_width):
            raise ValueError(f"default_width_per_col {default_width_per_col} "
                             f"> max_width {max_width}")
    if default_height_per_row is not None:
        default_height_per_row = float(default_height_per_row)
        if (not isfinite(default_height_per_row)) \
                or (default_height_per_row <= 0.1) \
                or (default_height_per_row >= 10000):
            raise ValueError(
                f"invalid default_height_per_row {default_height_per_row}")
    if max_height is not None:
        max_height = float(max_height)
        if (not isfinite(max_height)) \
                or (max_height <= 0.1) or (max_height >= 10000):
            raise ValueError(f"invalid max_width {max_height}")
        if (default_height_per_row is not None) \
                and (max_height < default_height_per_row):
            raise ValueError(
                f"max_height {max_height} < "
                f"default_height_per_row {default_height_per_row}")

    # setting up the default dimensions
    if default_height_per_row is None:
        default_height_per_row = 1.95
        if max_height is not None:
            default_height_per_row = min(default_height_per_row, max_height)
    if max_height is None:
        max_height = max_rows * default_height_per_row
    if default_width_per_col is None:
        default_width_per_col = 8.6
        if max_width is not None:
            default_width_per_col = min(default_width_per_col, max_width)
    if max_width is None:
        max_width = default_width_per_col

    # How many plots do we need at least? Well, if we can put
    # max_items_per_plot items into each plot and have items many items, then
    # we need ceil(items / max_items_per_plot) many. This is what we compute
    # here.
    min_plots: Final[int] = (items // max_items_per_plot) \
        + min(1, items % max_items_per_plot)
    # The maximum conceivable number of plots would be items, so this we do
    # not need to compute.
    plots_i: Final[int] = 6
    rows_i: Final[int] = plots_i + 1
    cols_i: Final[int] = rows_i + 1
    height_i: Final[int] = cols_i + 1
    width_i: Final[int] = height_i + 1
    plots_per_row_i: Final[int] = width_i + 1
    chunks_i: Final[int] = plots_per_row_i + 1
    best: Tuple[float,  # the plots-per-row std
                float,  # the items-per-plot std
                float,  # the overall area of the figure
                float,  # the std deviation of (w, h*golden_ratio),
                float,  # -plot area
                float,  # the std deviation of (w, h*golden_ratio) per plot
                int,  # the number of plots
                int,  # the number of rows
                int,  # the number of cols
                float,  # the figure height
                float,  # the figure width
                List[int],  # the plots per row
                List[int]] = \
        (1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0,
         1 << 62, 1 << 62, 1 << 62, 1000000.0, 1000000.0, [0], [0])

    # Now we simply try all valid combinations.
    for rows in range(min_rows, max_rows + 1):
        if rows > items:
            continue  # more rows than items
        for cols in range(min_cols, max_cols + 1):
            # We need cols plots per row, except for the last row, where we
            # could only put a single plot if we wanted.
            min_plots_for_config: int = 1 + ((rows - 1) * cols)
            if min_plots_for_config > items:
                continue  # if this exceeds the maximum number of plots, skip
            # compute the maximum plots we can get in this config
            max_plots_per_config: int = rows * cols
            if max_plots_per_config < min_plots:
                continue  # not enough plots possible

            # now we can iterate over the number of plots we can provide
            for plots in range(max(min_plots_for_config, min_plots),
                               min(max_plots_per_config, items) + 1):
                # We compute the figure width. If we have multiple columns,
                # we try to add 2% plus 0.02" as breathing space per column.
                fig_width = min(max_width, ((1.02 * default_width_per_col)
                                            + 0.02) * cols)
                # The width of a single plot is then computed as follows:
                plot_width = fig_width / cols
                if plot_width < 0.9:
                    continue  # the single plots would be too small
                # We compute the overall figure height. Again we add breathing
                # space if we have multiple rows.
                fig_height = min(max_height, ((1.02 * default_height_per_row)
                                              + 0.02) * rows)
                plot_height = fig_height / rows
                if plot_height < 0.9:
                    continue  # the single plots would be too small
                # So dimension-wise, the plot sizes are OK.
                # How about the distribution of items? We put the rows with
                # plots at the top and the plots with more items at the end.
                plot_distr = __divide_evenly(plots, rows, reverse=False)
                item_distr = __divide_evenly(items, plots, reverse=True)

                current: Tuple[float, float, float, float, float, float,
                               int, int, int, float, float,
                               List[int], List[int]] = (
                    st.stdev(plot_distr) if rows > 1 else 0,
                    st.stdev(item_distr) if plots > 1 else 0,
                    fig_height * fig_width,  # the area of the figure
                    st.stdev([fig_width, fig_height * __GOLDEN_RATIO]),
                    -plot_width * plot_height,  # -plot area
                    st.stdev([plot_width, plot_height * __GOLDEN_RATIO]),
                    plots,  # the number of plots
                    rows,  # the number of rows
                    cols,  # the number of cols
                    fig_height,  # the figure height
                    fig_width,  # the figure width
                    plot_distr,  # the plots per row
                    item_distr)  # the data chunks per plot
                if ((current[plots_i] + 1)  # type: ignore
                    < best[plots_i]) or ((  # type: ignore
                        current[plots_i]   # type: ignore
                        <= (best[plots_i] + 1))  # type: ignore
                        and (current < best)):  # type: ignore
                    best = current

    n_plots: Final[int] = best[plots_i]  # type: ignore
    if n_plots > items:
        raise ValueError(
            f"Could not place {items} in {min_cols}..{max_cols} columns "
            f"at {min_rows}..{max_rows} rows with at most "
            f"{max_items_per_plot} items per plot for a max_width={max_width}"
            f" and max_height={max_height}.")

    # create the figure of the computed dimensions
    figure: Final[Figure] = create_figure(
        width=best[width_i], height=best[height_i],  # type: ignore
        dpi=dpi, **kwargs)
    if n_plots <= 1:  # if there is only one plot, we are done here
        return figure, tuple([(figure, 0, items, 0, 0, 0)])

    # if there are multiple plots, we need to generate them
    allfigs: List[Tuple[Union[SubplotBase, Figure],
                        int, int, int, int, int]] = []
    index: int = 0
    chunk_start: int = 0
    nrows: Final[int] = best[rows_i]  # type: ignore
    ncols: Final[int] = best[cols_i]  # type: ignore
    chunks: Final[List[int]] = best[chunks_i]  # type: ignore
    plots_per_row: Final[List[int]] = best[plots_per_row_i]  # type: ignore
    for i in range(nrows):
        for j in range(plots_per_row[i]):
            chunk_next = chunk_start + chunks[index]
            allfigs.append((figure.add_subplot(nrows, ncols,
                                               (i * ncols) + j + 1),
                            chunk_start, chunk_next, i, j, index))
            chunk_start = chunk_next
            index += 1
    return figure, tuple(allfigs)


def save_figure(fig: Figure,
                file_name: str = "figure",
                dir_name: str = ".",
                formats: Union[str, Iterable[str]] = "svg") -> List[Path]:
    """
    Store the given figure in files of the given formats and dispose it.

    :param fig: the figure to save
    :param file_name: the basic file name
    :param dir_name: the directory name
    :param formats: the file format(s)
    :return: a list of files
    """
    if not isinstance(fig, Figure):
        raise type_error(fig, "figure", Figure)
    if not isinstance(file_name, str):
        raise type_error(file_name, "file_name", str)
    if len(file_name) <= 0:
        raise ValueError(f"Invalid filename '{file_name}'.")
    if not isinstance(dir_name, str):
        raise type_error(dir_name, "dir_name", str)
    if len(dir_name) <= 0:
        raise ValueError(f"Invalid dirname '{dir_name}'.")
    if isinstance(formats, str):
        formats = [formats]
    if not isinstance(formats, Iterable):
        raise type_error(formats, "formats", Iterable)

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
            raise type_error(fmt, "element of formats", str)
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
              z_order: Optional[float] = None,
              font: Union[None, str, Callable] =
              lambda: Lang.current().font()) -> None:
    """
    Put a label text box near an axis.

    :param axes: the axes to add the label to
    :param text: the text to place
    :param x: the location along the x-axis: `0` means left,
        `0.5` means centered, `1` means right
    :param y: the location along the x-axis: `0` means bottom,
        `0.5` means centered, `1` means top
    :param font_size: the font size
    :param may_rotate_text: should we rotate the text by 90° if that
        makes sense (`True`) or always keep it horizontally (`False`)
    :param z_order: an optional z-order value
    :param font: the font to use
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
    if z_order is not None:
        args["zorder"] = z_order

    if may_rotate_text and (len(text) > 2):
        args["rotation"] = 90

    if callable(font):
        font = font()
    if font is not None:
        if not isinstance(font, str):
            raise type_error(font, "font", str)
        args['fontname'] = font

    axes.annotate(**args)


def label_axes(axes: Axes,
               x_label: Optional[str] = None,
               x_label_inside: bool = True,
               x_label_location: float = 0.5,
               y_label: Optional[str] = None,
               y_label_inside: bool = True,
               y_label_location: float = 1,
               font_size: float = pd.importance_to_font_size(0),
               z_order: Optional[float] = None) -> None:
    """
    Put labels on a figure.

    :param axes: the axes to add the label to
    :param x_label: a callable returning the label for
        the x-axis, a label string, or `None` if no label should be put
    :param x_label_inside: put the x-axis label inside the plot (so that
        it does not consume additional vertical space)
    :param x_label_location: the location of the x-axis label if it is
        placed inside the plot area
    :param y_label: a callable returning the label for
        the y-axis, a label string, or `None` if no label should be put
    :param y_label_inside: put the xyaxis label inside the plot (so that
        it does not consume additional horizontal space)nal vertical space)
    :param y_label_location: the location of the y-axis label if it is
        placed inside the plot area
    :param font_size: the font size to use
    :param z_order: an optional z-order value
    """
    # put the label on the x-axis, if any
    if x_label is not None:
        if not isinstance(x_label, str):
            raise type_error(x_label, "x_label", str)
        if len(x_label) > 0:
            if x_label_inside:
                label_box(axes, text=x_label, x=x_label_location, y=0,
                          font_size=font_size, z_order=z_order)
            else:
                axes.set_xlabel(x_label, fontsize=font_size)

    # put the label on the y-axis, if any
    if y_label is not None:
        if not isinstance(y_label, str):
            raise type_error(y_label, "y_label", str)
        if len(y_label) > 0:
            if y_label_inside:
                label_box(axes, text=y_label, x=0, y=y_label_location,
                          font_size=font_size, may_rotate_text=True,
                          z_order=z_order)
            else:
                axes.set_ylabel(y_label, fontsize=font_size)


def get_axes(figure: Union[Axes, SubplotBase, Figure]) -> Axes:
    """
    Obtain the axes from a figure or axes object.

    :param figure: the figure
    :return: the Axes
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
    raise TypeError(
        f"Cannot get Axes of object of type {type_name_of(figure)}.")


def get_renderer(figure: Union[SubplotBase, Axes, Figure]) -> RendererBase:
    """
    Get a renderer that can be used for determining figure element sizes.

    :param figure: the figure element
    :return: the renderer
    """
    if isinstance(figure, (Axes, SubplotBase)):
        figure = figure.figure
    if not isinstance(figure, Figure):
        raise type_error(figure, "figure", Figure)
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

    :param cm: the cm value
    :return: the value in inch
    """
    if not isinstance(cm, int):
        if not isinstance(cm, float):
            raise type_error(cm, "cm", (int, float))
        if not isfinite(cm):
            raise ValueError(f"cm must be finite, but is {cm}.")
    res: float = cm / 2.54
    if not isfinite(res):
        raise ValueError(f"Conversation {cm} cm to inch "
                         f"must be finite, but is {res}.")
    return res


#: the color attributes
__COLOR_ATTRS: Final[Tuple[Tuple[bool, bool, str], ...]] = \
    ((True, True, "get_label"), (False, True, "label"),
     (False, True, "_label"), (True, False, "get_color"),
     (False, False, "color"), (False, False, "_color"),
     (False, False, "edgecolor"), (False, False, "_edgecolor"),
     (False, False, "markeredgecolor"), (False, False, "_markeredgecolor"))


def get_label_colors(handles: Iterable[Artist], color_map: Optional[Dict[
                     str, Union[Tuple[float, ...], str]]] = None,
                     default_color: Union[Tuple[float, ...], str]
                     = pd.COLOR_BLACK) -> List[Union[Tuple[float, ...], str]]:
    """
    Get a list with label colors from a set of artists.

    :param handles: the handles
    :param color_map: an optional color map
    :param default_color: the default color
    :returns: a list of label colors
    """
    if not isinstance(handles, Iterable):
        raise type_error(handles, "handles", Iterable)

    def __get_color(a: Artist, colmap=color_map, defcol=default_color):
        if not isinstance(a, Artist):
            raise type_error(a, "artist", Artist)

        for acall, astr, aname in __COLOR_ATTRS:
            if hasattr(a, aname):
                val = getattr(a, aname)
                if val is None:
                    continue
                if acall:
                    val = val()
                if val is None:
                    continue
                if astr:
                    if colmap:
                        if val in colmap:
                            val = colmap[val]
                        else:
                            continue
                    else:
                        continue
                if val is None:
                    continue
                if val != defcol:
                    return val
        return defcol

    return [__get_color(aa) for aa in handles]
