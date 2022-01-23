"""Violin plots for end results."""
from math import isfinite
from typing import List, Dict, Final, Callable, Iterable, Union, \
    Tuple, Set, Optional, Any

import numpy as np
from matplotlib.axes import Axes  # type: ignore
from matplotlib.collections import PolyCollection  # type: ignore
from matplotlib.figure import Figure, SubplotBase  # type: ignore

import moptipy.evaluation.plot_defaults as pd
import moptipy.evaluation.plot_utils as pu
from moptipy.evaluation._utils import _try_div2
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import F_NAME_RAW, F_NAME_SCALED, \
    F_NAME_NORMALIZED
from moptipy.evaluation.end_results import EndResult
from moptipy.utils.log import logger
from moptipy.utils.logging import KEY_LAST_IMPROVEMENT_FE, \
    KEY_LAST_IMPROVEMENT_TIME_MILLIS, KEY_TOTAL_FES, \
    KEY_TOTAL_TIME_MILLIS

#: the permitted dimensions
__PERMITTED_DIMENSIONS: Final[Tuple[str, str, str, str, str, str, str]] = \
    (F_NAME_RAW, F_NAME_SCALED, F_NAME_NORMALIZED,
     KEY_LAST_IMPROVEMENT_FE, KEY_TOTAL_FES,
     KEY_LAST_IMPROVEMENT_TIME_MILLIS, KEY_TOTAL_TIME_MILLIS)


def plot_end_violin(
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
        plot_arith_mean: bool = True,
        plot_median: bool = True,
        plot_q25: bool = True,
        plot_q75: bool = True,
        distinct_markers_func: Callable = pd.distinct_markers,
        goal_f: Union[int, float, Callable] = lambda g: g.goal_f) -> None:
    """
    Plot a set of Ecdf functions into one chart.

    :param end_results: the iterable of end results
    :param Union[SubplotBase, Figure] figure: the figure to plot in
    :param str dimension: the dimension to display
    :param Union[moptipy.evaluation.AxisRanger, Callable] y_axis: the y_axis
    :param Callable distinct_colors_func: the function returning the palette
    :param Callable importance_to_line_width_func: the function converting
        importance values to line widths
    :param Callable importance_to_font_size_func: the function converting
        importance values to font sizes
    :param bool ygrid: should we have a grid along the y-axis?
    :param bool xgrid: should we have a grid along the x-axis?
    :param bool plot_arith_mean: should we plot the arithmetic mean?
    :param bool plot_median: should we plot the median?
    :param bool plot_q25: should we plot the 25% quantile?
    :param bool plot_q75: should we plot the 75% quantile?
    :param Callable distinct_markers_func: the function for creating the
        distinct markers for the statistics
    :param Union[int, float, Callable] goal_f: the goal objective value for
        normalized or standardized f display. Can be a constant or a callable
        applied to the individual EndResult records.
    """
    if not isinstance(dimension, str):
        raise TypeError(f"dimension must be str, but is {type(dimension)}.")
    if dimension not in __PERMITTED_DIMENSIONS:
        raise ValueError(f"dimension is '{dimension}', but must be one of "
                         f"{__PERMITTED_DIMENSIONS}.")
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

        value: Union[int, float]
        if dimension == F_NAME_RAW:
            value = res.best_f
        elif dimension in (F_NAME_SCALED, F_NAME_NORMALIZED):
            goal = goal_f(res) if callable(goal_f) else goal_f
            if not isinstance(goal, (int, float)):
                raise TypeError(
                    f"goal must be int or float, but goal {goal} obtained"
                    f"from {goal_f} for {res} is {type(goal)}.")
            if not isfinite(goal):
                raise ValueError(f"goal must be finite, but is {goal}.")
            if goal == 0:
                raise ValueError(f"goal must not be 0, but is {goal}.")
            if dimension == F_NAME_SCALED:
                value = _try_div2(res.best_f, goal)
            else:
                value = _try_div2(res.best_f - goal, goal)
        elif dimension == KEY_TOTAL_FES:
            value = res.total_fes
        elif dimension == KEY_LAST_IMPROVEMENT_FE:
            value = res.last_improvement_fe
        elif dimension == KEY_TOTAL_TIME_MILLIS:
            value = res.total_time_millis
        elif dimension == KEY_LAST_IMPROVEMENT_TIME_MILLIS:
            value = res.last_improvement_time_millis
        else:
            raise ValueError(f"huh? dimension is {dimension}??")
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
    algorithms: Final[Tuple[str, ...]] = tuple(sorted(algo_set))
    logger(f"- {n_algorithms} algorithms ({algorithms}) "
           f"and {n_instances} instances ({data.keys()}).")

    # compile the data
    inst_algos: List[Tuple[str, List[str]]] = []
    plot_data: List[List[Union[int, float]]] = []
    plot_algos: List[str] = []
    for inst in sorted(data.keys()):
        per_inst_data = data[inst]
        algo_names: List[str] = sorted(per_inst_data.keys())
        plot_algos.extend(algo_names)
        inst_algos.append((inst, algo_names))
        for algo in algo_names:
            inst_algo_data = per_inst_data[algo]
            inst_algo_data.sort()
            plot_data.append(inst_algo_data)

    # compute the violin positions
    n_violins: Final[int] = len(plot_data)
    if n_violins < max(n_instances, n_algorithms):
        raise ValueError(f"Huh? {n_violins}, {n_instances}, {n_algorithms}")
    violin_positions: Final[Tuple[int, ...]] = \
        tuple(range(1, len(plot_data) + 1))

    # Now we got all instances and all algorithms and know the axis ranges.
    font_size_0: Final[float] = importance_to_font_size_func(0)

    # set up the graphics area
    axes: Final[Axes] = pu.get_axes(figure)
    axes.tick_params(axis="y", labelsize=font_size_0)
    axes.tick_params(axis="x", labelsize=font_size_0)

    # draw the grid
    grid_lwd: Optional[Union[int, float]] = None
    if ygrid:
        grid_lwd = importance_to_line_width_func(-1)
        axes.grid(axis="y", color=pd.GRID_COLOR, linewidth=grid_lwd)

    x_axis: Final[AxisRanger] = AxisRanger(
        chosen_min=0.5, chosen_max=violin_positions[-1] + 0.5)

    # compute the labels for the x-axis
    labels_str: List[str] = []
    labels_x: List[float] = []
    counter: int = 0
    needs_legend: bool = False

    if n_instances > 1:
        # use only the instances as labels
        for key in inst_algos:
            current = counter
            counter += len(key[1])
            labels_str.append(key[0])
            labels_x.append(0.5 * (violin_positions[current]
                                   + violin_positions[counter - 1]))
        needs_legend = (n_algorithms > 1)
    elif n_algorithms > 1:
        # only use algorithms as key
        for key in inst_algos:
            for algo in key[1]:
                labels_str.append(algo)
                labels_x.append(violin_positions[counter])
                counter += 1

    # manually add x grid lines between instances
    if xgrid and (n_instances > 1) and (n_algorithms > 1):
        if not grid_lwd:
            grid_lwd = importance_to_line_width_func(-1)
        counter = 0
        for key in inst_algos:
            if counter > 0:
                axes.axvline(x=counter + 0.5,
                             color=pd.GRID_COLOR,
                             linewidth=grid_lwd)
            counter += len(key[1])

    y_axis.apply(axes, "y")
    x_axis.apply(axes, "x")

    violins: Final[Dict[str, Any]] = axes.violinplot(
        dataset=plot_data, positions=violin_positions, vert=True,
        widths=2 / 3, showmeans=False, showextrema=False, showmedians=False)

    # fix the algorithm colors
    colors: Final[Tuple[Any]] = distinct_colors_func(n_algorithms)
    algo_colors: Dict[str, Tuple[float, float, float]] = {}
    for i, algo in enumerate(algorithms):
        algo_colors[algo] = colors[i]

    bodies: Final[PolyCollection] = violins["bodies"]
    counter = 0
    for key in inst_algos:
        for algo in key[1]:
            bd = bodies[counter]
            color = algo_colors[algo]
            bd.set_edgecolor('none')
            bd.set_facecolor(color)
            bd.set_alpha(1)
            counter += 1
            bd.set_zorder(100 + counter)

    marker_cmd: List[Callable] = []
    marker_names: List[str] = []
    if plot_q25:
        marker_cmd.append(lambda x: np.percentile(x, 25))
        marker_names.append("q25")
    if plot_median:
        marker_cmd.append(np.median)
        marker_names.append("median")
    if plot_arith_mean:
        marker_cmd.append(np.mean)
        marker_names.append("mean")
    if plot_q75:
        marker_cmd.append(lambda x: np.percentile(x, 75))
        marker_names.append("q75")

    if len(marker_cmd) > 0:
        counter = 0
        markers: Final[Tuple[str, ...]] = \
            distinct_markers_func(len(marker_cmd))
        for key in inst_algos:
            for algo in key[1]:
                pdata = plot_data[counter]
                color = pd.text_color_for_background(algo_colors[algo])
                for j, marker in enumerate(marker_cmd):
                    axes.scatter(x=violin_positions[counter],
                                 y=marker(pdata),
                                 color=color,
                                 marker=markers[j],
                                 zorder=200 + (20 * counter) + j)
                counter += 1

    if needs_legend:
        axes.legend("x")

    logger(f"done plotting {n_violins} end violins.")
