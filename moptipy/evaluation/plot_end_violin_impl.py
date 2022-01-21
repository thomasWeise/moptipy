"""Violin plots for end results."""
from math import isfinite
from typing import List, Dict, Final, Callable, Iterable, Union, \
    Tuple

from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure, SubplotBase  # type: ignore

import moptipy.evaluation.plot_defaults as pd
import moptipy.evaluation.plot_utils as pu
from moptipy.evaluation._utils import _try_div2
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import F_NAME_RAW, F_NAME_SCALED, \
    F_NAME_NORMALIZED
from moptipy.evaluation.base import get_algorithm
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.styler import Styler
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
    :param Union[int, float, Callable] goal_f: the goal objective value for
        normalized or standardized f display. Can be a constant or a callable
        applied to the individual EndResult records.
    """
    if not isinstance(dimension, str):
        raise TypeError(f"dimension must be str, but is {type(dimension)}.")
    if dimension not in __PERMITTED_DIMENSIONS:
        raise ValueError(f"dimension is '{dimension}', but must be one of "
                         f"{__PERMITTED_DIMENSIONS}.")

    # instance -> algorithm -> values
    data: Dict[str, Dict[str, List[Union[int, float]]]] = {}
    algorithms: Final[Styler] = Styler(get_algorithm, "all algos", 0)

    if callable(y_axis):
        y_axis = y_axis(dimension)
    if not isinstance(y_axis, AxisRanger):
        raise TypeError(f"y_axis for {dimension} must be AxisRanger, "
                        f"but is {type(y_axis)}.")

    # We now collect instances, the algorithms, and the measured values.
    for res in end_results:
        if not isinstance(res, EndResult):
            raise TypeError(
                "all violin plot elements must be instances of EndResult, "
                f"but encountered an instance of {type(res)}.")

        inst = res.instance
        d: Dict[str, List[Union[int, float]]]
        if inst not in data:
            d = {}
            data[inst] = d
        else:
            d = data[inst]

        algorithms.add(res)
        ad: List[Union[int, float]]
        if res.algorithm not in d:
            ad = []
            d[res.algorithm] = ad
        else:
            ad = d[res.algorithm]

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
        ad.append(value)
        y_axis.register_value(value)
    del inst, d, ad

    if len(data) <= 0:
        raise ValueError("Data cannot be empty.")

    # compile the algorithm data
    algorithms.compile()
    if algorithms.count > 1:
        algorithms.set_fill_color(distinct_colors_func)
        algorithms.set_line_color(distinct_colors_func)

    # compile the data
    keys: List[Tuple[str, List[str]]] = []
    plot_data: List[List[Union[int, float]]] = []
    for inst in sorted(data.keys()):
        dd: Dict[str, List[Union[int, float]]] = data[inst]
        cc: List[str] = []
        keys.append((inst, cc))
        for algo in sorted(dd.keys()):
            cc.append(algo)
            ll = dd[algo]
            ll.sort()
            plot_data.append(ll)
    del data, dd, cc, ll

    # Now we got all instances and all algorithms and know the axis ranges.
    font_size_0: Final[float] = importance_to_font_size_func(0)

    # compute the violin positions
    positions = list(range(1, len(plot_data) + 1))

    # set up the graphics area
    axes: Final[Axes] = pu.get_axes(figure)
    axes.tick_params(axis="y", labelsize=font_size_0)

    # draw the grid
    if ygrid:
        grid_lwd = importance_to_line_width_func(-1)
        axes.grid(axis="y", color=pd.GRID_COLOR, linewidth=grid_lwd)

    # compute the labels for the x-axis
    labels: List[str] = []
    labels_x: List[float] = []
    use_instances_on_x: bool = len(keys) > 1
    use_algorithms_on_x: bool = algorithms.count > 1
    needs_legend: bool
    if (len(plot_data) > 6) and use_algorithms_on_x and use_instances_on_x:
        use_algorithms_on_x = False
        needs_legend = True
    else:
        needs_legend = not (use_algorithms_on_x or use_instances_on_x)

    total_violins: int = 0
    if use_instances_on_x:
        if use_algorithms_on_x:
            # use both instances and algorithms as labels
            for key in keys:
                for algo in key[1]:
                    labels.append(f"{key[0]}\n{algo}")
                    labels_x.append(positions[total_violins])
                    total_violins += 1
        else:
            # use only instances as labels
            for key in keys:
                cur_violins = total_violins
                total_violins += len(key[1])
                labels.append(key[0])
                labels_x.append(0.5 * (positions[cur_violins]
                                       + positions[total_violins - 1]))
    elif use_algorithms_on_x:
        # only use algorithms as key
        for key in keys:
            for algo in key[1]:
                labels.append(algo)
                labels_x.append(positions[total_violins])
                total_violins += 1

    axes.violinplot(dataset=plot_data,
                    positions=positions,
                    vert=True,
                    widths=0.5,
                    showmeans=True,
                    showextrema=True,
                    showmedians=True)
    y_axis.apply(axes, "y")

    if needs_legend:
        axes.legend("x")
