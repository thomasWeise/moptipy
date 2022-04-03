"""Violin plots for end results."""
from typing import List, Dict, Final, Callable, Iterable, Union, \
    Tuple, Set, Optional, Any

import matplotlib.collections as mc  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure, SubplotBase  # type: ignore
from matplotlib.lines import Line2D  # type: ignore

import moptipy.utils.plot_defaults as pd
import moptipy.utils.plot_utils as pu
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import F_NAME_SCALED
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.lang import Lang
from moptipy.utils.log import logger


def plot_end_results(
        end_results: Iterable[EndResult],
        figure: Union[SubplotBase, Figure],
        dimension: str = F_NAME_SCALED,
        y_axis: Union[AxisRanger, Callable] = AxisRanger.for_axis,
        distinct_colors_func: Callable = pd.distinct_colors,
        importance_to_line_width_func: Callable =
        pd.importance_to_line_width,
        importance_to_font_size_func: Callable =
        pd.importance_to_font_size,
        ygrid: bool = True,
        xgrid: bool = True,
        xlabel: Union[None, str, Callable] = Lang.translate,
        xlabel_inside: bool = True,
        ylabel: Union[None, str, Callable] = Lang.translate,
        ylabel_inside: bool = True,
        legend_pos: str = "best",
        instance_sort_key: Callable = lambda x: x,
        algorithm_sort_key: Callable = lambda x: x) -> None:
    """
    Plot a set of end result boxes/violins functions into one chart.

    In this plot, we combine two visualizations of data distributions: box
    plots in the foreground and violin plots in the background.

    The box plots show you the median, the 25% and 75% quantiles, the 95%
    confidence interval around the median (as notches), the 5% and 95%
    quantiles (as whiskers), the arithmetic mean (as triangle), and the
    outliers on both ends of the spectrum. This allows you also to compare
    data from different distributions rather comfortably, as you can, e.g.,
    see whether the confidence intervals overlap.

    The violin plots in the background are something like smoothed-out,
    vertical, and mirror-symmetric histograms. They give you a better
    impression about shape and modality of the distribution of the results.

    :param end_results: the iterable of end results
    :param figure: the figure to plot in
    :param dimension: the dimension to display
    :param y_axis: the y_axis ranger
    :param distinct_colors_func: the function returning the palette
    :param importance_to_line_width_func: the function converting importance
        values to line widths
    :param importance_to_font_size_func: the function converting importance
        values to font sizes
    :param ygrid: should we have a grid along the y-axis?
    :param xgrid: should we have a grid along the x-axis?
    :param xlabel: a callable returning the label for the x-axis, a label
        string, or `None` if no label should be put
    :param xlabel_inside: put the x-axis label inside the plot (so that
        it does not consume additional vertical space)
    :param ylabel: a callable returning the label for the y-axis, a label
        string, or `None` if no label should be put
    :param ylabel_inside: put the y-axis label inside the plot (so that it
        does not consume additional horizontal space)
    :param legend_pos: the legend position
    :param instance_sort_key: the sort key function for instances
    :param algorithm_sort_key: the sort key function for algorithms
    """
    getter: Final[Callable[[EndResult], Union[int, float]]] \
        = EndResult.getter(dimension)
    logger(f"now plotting end violins for dimension {dimension}.")

    if callable(y_axis):
        y_axis = y_axis(dimension)
    if not isinstance(y_axis, AxisRanger):
        raise TypeError(f"y_axis for {dimension} must be AxisRanger, "
                        f"but is {type(y_axis)}.")

    # instance -> algorithm -> values
    data: Dict[str, Dict[str, List[Union[int, float]]]] = {}
    algo_set: Set[str] = set()

    # We now collect instances, the algorithms, and the measured values.
    for res in end_results:
        if not isinstance(res, EndResult):
            raise TypeError(
                "all violin plot elements must be instances of EndResult, "
                f"but encountered an instance of {type(res)}.")

        algo_set.add(res.algorithm)

        per_inst_data: Dict[str, List[Union[int, float]]]
        if res.instance not in data:
            data[res.instance] = per_inst_data = {}
        else:
            per_inst_data = data[res.instance]
        inst_algo_data: List[Union[int, float]]
        if res.algorithm not in per_inst_data:
            per_inst_data[res.algorithm] = inst_algo_data = []
        else:
            inst_algo_data = per_inst_data[res.algorithm]

        value: Union[int, float] = getter(res)
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"value must be int or float, but is {type(value)}.")
        inst_algo_data.append(value)
        y_axis.register_value(value)

    # We now know the number of instances and algorithms and have the data in
    # the hierarchical structure instance->algorithms->values.
    n_instances: Final[int] = len(data)
    n_algorithms: Final[int] = len(algo_set)
    if (n_instances <= 0) or (n_algorithms <= 0):
        raise ValueError("Data cannot be empty but found "
                         f"{n_instances} and {n_algorithms}.")
    algorithms: Final[Tuple[str, ...]] = \
        tuple(sorted(algo_set, key=algorithm_sort_key))
    logger(f"- {n_algorithms} algorithms ({algorithms}) "
           f"and {n_instances} instances ({data.keys()}).")

    # compile the data
    inst_algos: List[Tuple[str, List[str]]] = []
    plot_data: List[List[Union[int, float]]] = []
    plot_algos: List[str] = []
    for inst in sorted(data.keys(), key=instance_sort_key):
        per_inst_data = data[inst]
        algo_names: List[str] = sorted(per_inst_data.keys())
        plot_algos.extend(algo_names)
        inst_algos.append((inst, algo_names))
        for algo in algo_names:
            inst_algo_data = per_inst_data[algo]
            inst_algo_data.sort()
            plot_data.append(inst_algo_data)

    # compute the violin positions
    n_bars: Final[int] = len(plot_data)
    if n_bars < max(n_instances, n_algorithms):
        raise ValueError(f"Huh? {n_bars}, {n_instances}, {n_algorithms}")
    bar_positions: Final[Tuple[int, ...]] = \
        tuple(range(1, len(plot_data) + 1))

    # Now we got all instances and all algorithms and know the axis ranges.
    font_size_0: Final[float] = importance_to_font_size_func(0)

    # set up the graphics area
    axes: Final[Axes] = pu.get_axes(figure)
    axes.tick_params(axis="y", labelsize=font_size_0)
    axes.tick_params(axis="x", labelsize=font_size_0)

    zorder: int = 0

    # draw the grid
    grid_lwd: Optional[Union[int, float]] = None
    if ygrid:
        grid_lwd = importance_to_line_width_func(-1)
        zorder += 1
        axes.grid(axis="y", color=pd.GRID_COLOR, linewidth=grid_lwd,
                  zorder=zorder)

    x_axis: Final[AxisRanger] = AxisRanger(
        chosen_min=0.5, chosen_max=bar_positions[-1] + 0.5)

    # manually add x grid lines between instances
    if xgrid and (n_instances > 1) and (n_algorithms > 1):
        if not grid_lwd:
            grid_lwd = importance_to_line_width_func(-1)
        counter: int = 0
        for key in inst_algos:
            if counter > 0:
                zorder += 1
                axes.axvline(x=counter + 0.5,
                             color=pd.GRID_COLOR,
                             linewidth=grid_lwd,
                             zorder=zorder)
            counter += len(key[1])

    y_axis.apply(axes, "y")
    x_axis.apply(axes, "x")

    violin_width: Final[float] = 3 / 4
    zorder += 1
    violins: Final[Dict[str, Any]] = axes.violinplot(
        dataset=plot_data, positions=bar_positions, vert=True,
        widths=violin_width, showmeans=False, showextrema=False,
        showmedians=False)

    # fix the algorithm colors
    unique_colors: Final[Tuple[Any]] = distinct_colors_func(n_algorithms)
    algo_colors: Final[Dict[str, Tuple[float, float, float]]] = {}
    for i, algo in enumerate(algorithms):
        algo_colors[algo] = unique_colors[i]

    bodies: Final[mc.PolyCollection] = violins["bodies"]
    use_colors: Final[List[Tuple[float, float, float]]] = []
    counter = 0
    for key in inst_algos:
        for algo in key[1]:
            zorder += 1
            bd = bodies[counter]
            color = algo_colors[algo]
            use_colors.append(color)
            bd.set_edgecolor("none")
            bd.set_facecolor(color)
            bd.set_alpha(0.6666666666)
            counter += 1
            bd.set_zorder(zorder)

    zorder += 1
    boxes_bg: Final[Dict[str, Any]] = axes.boxplot(
        x=plot_data, positions=bar_positions, widths=violin_width,
        showmeans=True, patch_artist=False, notch=True, vert=True,
        whis=(5.0, 95.0), manage_ticks=False, zorder=zorder)
    zorder += 1
    boxes_fg: Final[Dict[str, Any]] = axes.boxplot(
        x=plot_data, positions=bar_positions, widths=violin_width,
        showmeans=True, patch_artist=False, notch=True, vert=True,
        whis=(5.0, 95.0), manage_ticks=False, zorder=zorder)

    for tkey in ("cmeans", "cmins", "cmaxes", "cbars", "cmedians",
                 "cquantiles"):
        if tkey in violins:
            violins[tkey].set_color("none")

    lwd_fg = importance_to_line_width_func(0)
    lwd_bg = importance_to_line_width_func(1)

    for bid, boxes in enumerate([boxes_bg, boxes_fg]):
        for tkey in ("boxes", "medians", "whiskers", "caps", "fliers",
                     "means"):
            if tkey not in boxes:
                continue
            polys: List[Line2D] = boxes[tkey]
            for line in polys:
                xdata = line.get_xdata(True)
                if len(xdata) <= 0:
                    line.remove()
                    continue
                index = int(max(xdata)) - 1
                thecolor = "white" if bid == 0 else use_colors[index]
                width = lwd_bg if bid == 0 else lwd_fg
                line.set_solid_joinstyle("round")
                line.set_solid_capstyle("round")
                line.set_color(thecolor)
                line.set_linewidth(width)
                line.set_markeredgecolor(thecolor)
                line.set_markerfacecolor("none")
                line.set_markeredgewidth(width)
                zorder = zorder + 1
                line.set_zorder(zorder)

    # compute the labels for the x-axis
    labels_str: List[str] = []
    labels_x: List[float] = []
    needs_legend: bool = False

    counter = 0
    if n_instances > 1:
        # use only the instances as labels
        for key in inst_algos:
            current = counter
            counter += len(key[1])
            labels_str.append(key[0])
            labels_x.append(0.5 * (bar_positions[current]
                                   + bar_positions[counter - 1]))
        needs_legend = (n_algorithms > 1)
    elif n_algorithms > 1:
        # only use algorithms as key
        for key in inst_algos:
            for algo in key[1]:
                labels_str.append(algo)
                labels_x.append(bar_positions[counter])
                counter += 1

    if labels_str:
        axes.set_xticks(ticks=labels_x, labels=labels_str, minor=False)
    else:
        axes.set_xticks([])

    zorder += 1
    pu.label_axes(axes=axes,
                  xlabel=xlabel("algorithm_on_instance")
                  if callable(xlabel) else xlabel,
                  xlabel_inside=xlabel_inside,
                  xlabel_location=1,
                  ylabel=ylabel(dimension) if callable(ylabel) else ylabel,
                  ylabel_inside=ylabel_inside,
                  ylabel_location=0.5,
                  font_size=font_size_0,
                  zorder=zorder)

    if needs_legend:
        handles: Final[List[Line2D]] = []

        for algo in algorithms:
            linestyle = pd.create_line_style()
            linestyle["label"] = algo
            legcol = algo_colors[algo]
            linestyle["color"] = legcol
            linestyle["markeredgecolor"] = legcol
            linestyle["xdata"] = []
            linestyle["ydata"] = []
            linestyle["linewidth"] = 6
            handles.append(Line2D(**linestyle))
        zorder += 1

        axes.legend(handles=handles, loc=legend_pos,
                    labelcolor=pu.get_label_colors(handles),
                    fontsize=font_size_0).set_zorder(zorder)

    logger(f"done plotting {n_bars} end result boxes.")
