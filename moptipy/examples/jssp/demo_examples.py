"""Some fixed demo codes for the JSSP examples."""
from typing import Final, List

import numpy as np
from matplotlib.figure import Figure  # type: ignore

from moptipy.evaluation.lang import Lang
from moptipy.evaluation.plot_utils import create_figure, save_figure, \
    cm_to_inch
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.examples.jssp.plot_gantt_chart_impl import plot_gantt_chart
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.utils.path import Path


def demo_instance() -> Instance:
    """
    Get the demo instance we use.

    :return: the demo instance
    :rtype: moptipy.examples.jssp.Instance
    """
    attr: Final[str] = "_res_"
    if not hasattr(demo_instance, attr):
        setattr(demo_instance, attr, Instance.from_resource("demo"))
    return getattr(demo_instance, attr)


def demo_search_space() -> PermutationsWithRepetitions:
    """
    Obtain an instance of the demo search space.

    :return: the demo search space
    :rtype: moptipy.examples.jssp.PermutationsWithRepetitions
    """
    attr: Final[str] = "_res_"
    if not hasattr(demo_search_space, attr):
        instance: Final[Instance] = demo_instance()
        setattr(demo_search_space, attr,
                PermutationsWithRepetitions(instance.jobs,
                                            instance.machines))
    return getattr(demo_search_space, attr)


def demo_point_in_search_space() -> np.ndarray:
    """
    Create a demo point in the search space.

    :return: the point
    :rtype: np.ndarray
    """
    space: Final[PermutationsWithRepetitions] = demo_search_space()
    res = space.create()
    np.copyto(res, [0, 2, 3, 2, 2, 3, 1, 1, 0, 3,
                    1, 3, 2, 1, 3, 2, 0, 1, 0, 0])
    space.validate(res)
    return res


def demo_solution_space() -> GanttSpace:
    """
    Obtain an instance of the demo solution space.

    :return: the demo solution space
    :rtype: moptipy.examples.jssp.GanttSpace
    """
    attr: Final[str] = "_res_"
    if not hasattr(demo_solution_space, attr):
        instance: Final[Instance] = demo_instance()
        setattr(demo_solution_space, attr, GanttSpace(instance))
    return getattr(demo_solution_space, attr)


def demo_encoding() -> OperationBasedEncoding:
    """
    Obtain an instance of the demo encoding.

    :return: the demo encoding
    :rtype: moptipy.examples.jssp.OperationBasedEncoding
    """
    attr: Final[str] = "_res_"
    if not hasattr(demo_encoding, attr):
        instance: Final[Instance] = demo_instance()
        setattr(demo_encoding, attr, OperationBasedEncoding(instance))
    return getattr(demo_encoding, attr)


def demo_solution() -> Gantt:
    """
    Create a demo solution.

    :return: the demo solution
    :rtype: moptipy.examples.jssp.Gantt
    """
    space: Final[GanttSpace] = demo_solution_space()
    result: Final[Gantt] = space.create()
    demo_encoding().map(demo_point_in_search_space(),
                        result)
    space.validate(result)
    return result


def demo_gantt_chart(dirname: str, filename: str =
                     "gantt_demo_without_makespan") -> List[Path]:
    """
    Plot the demo gantt chart.

    :param str dirname: the directory
    :param str filename: the file name
    """
    result: Final[List[Path]] = []
    the_dir: Final[Path] = Path.path(dirname)
    the_dir.ensure_dir_exists()

    for lang in Lang.all():
        lang.set_current()
        figure: Figure = create_figure(width=cm_to_inch(10))

        gantt = demo_solution()
        plot_gantt_chart(gantt=gantt,
                         figure=figure,
                         xlabel_inside=False,
                         ylabel_inside=False,
                         markers=None,
                         info=lambda g: Lang.current().format(
                             "gantt_info_no_ms", gantt=g))

        result.extend(save_figure(fig=figure,
                                  dir_name=the_dir,
                                  file_name=lang.filename(filename),
                                  formats="svg"))
    return result
