"""Plot a Gantt chart into one figure."""
from typing import Final, Callable, Union, Tuple

import numpy as np
from matplotlib.artist import Artist  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure, SubplotBase  # type: ignore
from matplotlib.patches import Rectangle  # type: ignore
from matplotlib.text import Text  # type: ignore
from matplotlib.ticker import MaxNLocator  # type: ignore

import moptipy.evaluation.plot_defaults as pd
import moptipy.evaluation.plot_utils as pu
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.examples.jssp.gantt import Gantt


def plot_gantt_chart(gantt: Gantt,
                     figure: Union[SubplotBase, Figure],
                     x_axis: Union[AxisRanger, Callable] =
                     lambda gantt: AxisRanger(chosen_min=0)) -> None:
    """
    Plot a Gantt chart.

    :param moptipy.examples.jssp.Gantt gantt: the gantt chart
    :param Union[SubplotBase, Figure] figure: the figure
    :param moptipy.evaluation.AxisRanger x_axis: the ranger for the x-axis
    """
    if not isinstance(gantt, Gantt):
        raise TypeError(f"gantt must be Gantt, but is {type(gantt)}.")
    axes: Final[Axes] = pu.get_axes(figure)

    # grab the data
    times: Final[np.ndarray] = gantt.times
    jobs: Final[int] = gantt.instance.jobs
    machines: Final[int] = gantt.instance.machines

    # Set up the x-axis range.
    if callable(x_axis):
        x_axis = x_axis(gantt)
    if not isinstance(x_axis, AxisRanger):
        raise TypeError(f"x_axis must be AxisRanger, but is {type(x_axis)}.")

    x_axis.register_array(times.flatten())
    x_axis.apply(axes, "x")

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
    font_size: Final = pd.importance_to_font_size(-1)

    # get the transforms needed to obtain text dimensions
    rend: Final = pu.get_renderer(axes)
    inv: Final = axes.transData.inverted()

    # plot the jobs
    for job in range(jobs):
        background = colors[job]
        foreground = pd.text_color_for_background(colors[job])
        jobstr = str(job)

        for machine in range(machines):
            # first plot the colored rectangle
            x_start = times[job][machine][0]
            x_end = times[job][machine][1]
            y_start = machine - bar_ofs

            axes.add_artist(Rectangle(
                xy=(x_start, y_start),
                width=(x_end - x_start),
                height=height,
                color=background,
                linewidth=0))

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
                          verticalalignment="center")
