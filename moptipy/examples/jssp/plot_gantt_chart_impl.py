"""Plot a Gantt chart into one figure."""
from typing import Final, Callable, Union, Tuple, Iterable, Dict, Optional

from matplotlib.artist import Artist  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure, SubplotBase  # type: ignore
from matplotlib.lines import Line2D  # type: ignore
from matplotlib.patches import Rectangle  # type: ignore
from matplotlib.text import Text  # type: ignore
from matplotlib.ticker import MaxNLocator  # type: ignore

import moptipy.utils.plot_defaults as pd
import moptipy.utils.plot_utils as pu
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.lang import Lang
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.makespan import makespan


def marker_lb(x: Gantt) -> Tuple[str, Union[int, float]]:
    """
    Compute the marker for the lower bound.

    :param x: the Gantt chart
    :return: the lower bound marker
    """
    return Lang.current()["lower_bound_short"],\
        x.instance.makespan_lower_bound


def marker_makespan(x: Gantt) -> Tuple[str, Union[int, float]]:
    """
    Compute the marker for the makespan.

    :param x: the Gantt chart
    :return: the makespan marker
    """
    return Lang.current()["makespan"], makespan(x)


#: the color for markers at the left end
__LEFT_END_MARK: Final[Tuple[float, float, float]] = (0.95, 0.02, 0.02)
#: the color for markers at the right end
__RIGHT_END_MARK: Final[Tuple[float, float, float]] = (0.02, 0.02, 0.95)
#: the color for markers in the middle
__MIDDLE_MARK: Final[Tuple[float, float, float]] = pd.COLOR_BLACK


def plot_gantt_chart(gantt: Gantt,
                     figure: Union[SubplotBase, Figure],
                     markers: Optional[Iterable[Union[
                         Tuple[str, Union[int, float]], Callable]]] =
                     (marker_lb,),
                     x_axis: Union[AxisRanger, Callable] =
                     lambda gantt: AxisRanger(chosen_min=0),
                     importance_to_line_width_func: Callable =
                     pd.importance_to_line_width,
                     importance_to_font_size_func: Callable =
                     pd.importance_to_font_size,
                     info: Union[None, str, Callable] = lambda g:
                     Lang.current().format("gantt_info", gantt=g),
                     xgrid: bool = False,
                     ygrid: bool = False,
                     xlabel: Union[None, str, Callable] = lambda g:
                     Lang.current()["time"],
                     xlabel_inside: bool = True,
                     ylabel: Union[None, str, Callable] = lambda g:
                     Lang.current()["machine"],
                     ylabel_inside: bool = True) -> None:
    """
    Plot a Gantt chart.

    :param gantt: the gantt chart
    :param figure: the figure
    :param markers: a set of markers
    :param x_axis: the ranger for the x-axis
    :param info: the optional info header
    :param importance_to_font_size_func: the function converting
        importance values to font sizes
    :param importance_to_line_width_func: the function converting
        importance values to line widths
    :param xgrid: should we have a grid along the x-axis?
    :param ygrid: should we have a grid along the y-axis?
    :param xlabel: a callable returning the label for
        the x-axis, a label string, or `None` if no label should be put
    :param xlabel_inside: put the x-axis label inside the plot (so that
        it does not consume additional vertical space)
    :param ylabel: a callable returning the label for
        the y-axis, a label string, or `None` if no label should be put
    :param ylabel_inside: put the y-axis label inside the plot (so that
        it does not consume additional horizontal space)
    """
    if not isinstance(gantt, Gantt):
        raise TypeError(f"gantt must be Gantt, but is {type(gantt)}.")
    axes: Final[Axes] = pu.get_axes(figure)

    # grab the data
    jobs: Final[int] = gantt.instance.jobs
    machines: Final[int] = gantt.instance.machines

    # Set up the x-axis range.
    if callable(x_axis):
        x_axis = x_axis(gantt)
    if not isinstance(x_axis, AxisRanger):
        raise TypeError(f"x_axis must be AxisRanger, but is {type(x_axis)}.")

    # Compute all the marks
    marks: Dict[Union[int, float], str] = {}
    if markers is not None:
        if not isinstance(markers, Iterable):
            raise TypeError(
                f"Expected markers to be Iterable, but got {type(markers)}.")
        for marker in markers:
            if callable(marker):
                marker = marker(gantt)
            if not marker:
                continue
            if isinstance(marker, tuple):
                name, val = marker
                if (not name) or (not val):
                    continue
            else:
                raise TypeError("marker must be tuple or "
                                f"callable, but is {type(marker)}")
            if not isinstance(name, str):
                raise TypeError(
                    f"marker name must be str, but is {type(name)}.")
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"marker must be int or float but is {type(val)}.")
            if val in marks:
                marks[val] = f"{marks[val]}/{name}"
            else:
                marks[val] = name
                x_axis.register_value(val)

    # Add x-axis data
    x_axis.register_array(gantt[:, 0, 1].flatten())  # register start times
    x_axis.register_array(gantt[:, -1, 2].flatten())  # register end times
    x_axis.apply(axes, "x")
    xmin, xmax = axes.get_xlim()

    # Set up the y-axis range.
    height: Final[float] = 0.7
    bar_ofs: Final[float] = height / 2
    y_min: Final[float] = -((7 * bar_ofs) / 6)
    y_max: Final[float] = (machines - 1) + ((7 * bar_ofs) / 6)

    axes.set_ylim(y_min, y_max)
    axes.set_ybound(y_min, y_max)
    axes.yaxis.set_major_locator(MaxNLocator(nbins="auto",
                                             integer=True))

    # get the color and font styles
    colors: Final[Tuple] = pd.distinct_colors(jobs)
    font_size: Final[float] = importance_to_font_size_func(-1)

    # get the transforms needed to obtain text dimensions
    rend: Final = pu.get_renderer(axes)
    inv: Final = axes.transData.inverted()

    # draw the grid
    if xgrid or ygrid:
        grid_lwd = importance_to_line_width_func(-1)
        if xgrid:
            axes.grid(axis="x", color=pd.GRID_COLOR, linewidth=grid_lwd)
        if ygrid:
            axes.grid(axis="y", color=pd.GRID_COLOR, linewidth=grid_lwd)

    zorder: int = 0

    # print the marker lines
    for val, _ in marks.items():
        axes.add_artist(Line2D(xdata=(val, val),
                               ydata=(y_min, y_max),
                               color=__LEFT_END_MARK if val <= xmin
                               else __RIGHT_END_MARK if val >= xmax
                               else __MIDDLE_MARK,
                               linewidth=2.0,
                               zorder=zorder))
        zorder += 1

    # plot the jobs
    for machine in range(machines):
        for jobi in range(jobs):
            job, x_start, x_end = gantt[machine, jobi, :]

            if x_end <= x_start:  # skip operations that take 0 time
                continue

            background = colors[job]
            foreground = pd.text_color_for_background(colors[job])
            jobstr = str(job)

            # first plot the colored rectangle
            y_start = machine - bar_ofs

            axes.add_artist(Rectangle(
                xy=(x_start, y_start),
                width=(x_end - x_start),
                height=height,
                color=background,
                linewidth=0,
                zorder=zorder))
            zorder += 1

            # Now we insert the job IDs, which is a bit tricky:
            # First, the rectangle may be too small to hold the text.
            # So we need to get the text bounding box size to compare it with
            # the rectangle's size.
            # If that fits, we can print the text.
            # Second, the printed text tends to be slightly off vertically,
            # even if we try to center it vertically. This is because fonts
            # can extend below their baseline which seems to be considered
            # during vertical alignment although job IDs are numbers and that
            # does not apply here. Therefore, we try to re-adjust the boxes
            # in a very, very crude way.
            xp: float = 0.5 * (x_start + x_end)
            yp = machine

            # Get the size of the text using a temporary text
            # that gets immediately deleted again.
            tmp: Text = axes.text(x=xp, y=yp, s=jobstr,
                                  fontsize=font_size,
                                  color=foreground,
                                  horizontalalignment="center",
                                  verticalalignment="baseline")
            bb_bl = inv.transform_bbox(tmp.get_window_extent(
                renderer=rend))
            Artist.set_visible(tmp, False)
            Artist.remove(tmp)
            del tmp

            if (bb_bl.width < 0.97 * (x_end - x_start)) and \
                    (bb_bl.height < (0.97 * height)):
                # OK, there is enough space. Let's re-compute the y
                # offset to do proper alignment using another temporary
                # text.
                tmp = axes.text(x=xp, y=yp, s=jobstr,
                                fontsize=font_size,
                                color=foreground,
                                horizontalalignment="center",
                                verticalalignment="bottom")
                bb_bt = inv.transform_bbox(tmp.get_window_extent(
                    renderer=rend))
                Artist.set_visible(tmp, False)
                Artist.remove(tmp)
                del tmp

                # Now we can really print the actual text with a more or less
                # nice vertical alignment.
                adj = bb_bl.y0 - bb_bt.y0
                if adj < 0:
                    yp += adj / 3

                axes.text(x=xp, y=yp, s=jobstr,
                          fontsize=font_size,
                          color=foreground,
                          horizontalalignment="center",
                          verticalalignment="center",
                          zorder=zorder)
                zorder += 1

    # print the marker labels
    bbox = {"boxstyle": 'round',
            "color": 'white',
            "fill": True,
            "linewidth": 0,
            "alpha": 0.9}
    y_mark: Final[float] = -0.1  # machines - 1 + (0.9 * bar_ofs) for top
    for val, name in marks.items():
        axes.annotate(text=f"{name}={val}",
                      xy=(val, y_mark),
                      xytext=(-4, -4),
                      verticalalignment="bottom",
                      horizontalalignment="right",
                      xycoords="data",
                      textcoords="offset points",
                      fontsize=font_size,
                      color=__LEFT_END_MARK if val <= xmin
                      else __RIGHT_END_MARK if val >= xmax
                      else __MIDDLE_MARK,
                      rotation=90,
                      bbox=bbox,
                      zorder=zorder)
        zorder += 1

    info_font_size: Final[float] = importance_to_font_size_func(0)
    pu.label_axes(axes=axes,
                  xlabel=xlabel(gantt) if callable(xlabel) else xlabel,
                  xlabel_inside=xlabel_inside,
                  xlabel_location=0.5,
                  ylabel=ylabel(gantt) if callable(ylabel) else ylabel,
                  ylabel_inside=ylabel_inside,
                  ylabel_location=0.5,
                  font_size=info_font_size,
                  zorder=zorder)
    zorder = zorder + 1

    if callable(info):
        info = info(gantt)
    if info is not None:
        if not isinstance(info, str):
            raise TypeError(f"info must be str, but is {type(info)}.")
        pu.label_box(axes=axes,
                     text=info,
                     x=0.5,
                     y=1,
                     font_size=importance_to_font_size_func(1),
                     may_rotate_text=False,
                     zorder=zorder)
