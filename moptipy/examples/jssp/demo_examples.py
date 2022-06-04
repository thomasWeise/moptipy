"""Some fixed demo codes for the JSSP examples."""
import sys
import urllib.request as rq
from typing import Final, List, Union, Callable, Iterable

import numpy as np
from matplotlib.figure import Figure  # type: ignore

from moptipy.examples.jssp.experiment import EXPERIMENT_INSTANCES
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance, \
    compute_makespan_lower_bound
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.examples.jssp.plot_gantt_chart_impl import plot_gantt_chart, \
    marker_lb, marker_makespan
from moptipy.spaces.permutations import Permutations
from moptipy.utils.console import logger
from moptipy.utils.lang import Lang
from moptipy.utils.path import Path, UTF8
from moptipy.utils.plot_utils import create_figure, save_figure, \
    cm_to_inch
from moptipy.utils.types import type_error


def demo_instance() -> Instance:
    """
    Get the demo instance we use.

    :return: the demo instance
    """
    attr: Final[str] = "_res_"
    if not hasattr(demo_instance, attr):
        setattr(demo_instance, attr, Instance.from_resource("demo"))
    return getattr(demo_instance, attr)


def demo_search_space() -> Permutations:
    """
    Obtain an instance of the demo search space.

    :return: the demo search space
    """
    attr: Final[str] = "_res_"
    if not hasattr(demo_search_space, attr):
        instance: Final[Instance] = demo_instance()
        setattr(demo_search_space, attr,
                Permutations.with_repetitions(instance.jobs,
                                              instance.machines))
    return getattr(demo_search_space, attr)


def demo_point_in_search_space(optimum: bool = False) -> np.ndarray:
    """
    Create a demo point in the search space.

    :param optimum: should we return the optimal solution?
    :return: the point
    """
    space: Final[Permutations] = demo_search_space()
    res = space.create()
    if optimum:
        np.copyto(res, [0, 1, 2, 3, 1, 0, 1, 2, 0, 1,
                        3, 2, 2, 3, 1, 3, 0, 2, 3, 0])
    else:
        np.copyto(res, [0, 2, 3, 2, 2, 3, 1, 1, 0, 3,
                        1, 3, 2, 1, 3, 2, 0, 1, 0, 0])
    space.validate(res)
    return res


def demo_solution_space() -> GanttSpace:
    """
    Obtain an instance of the demo solution space.

    :return: the demo solution space
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
    """
    attr: Final[str] = "_res_"
    if not hasattr(demo_encoding, attr):
        instance: Final[Instance] = demo_instance()
        setattr(demo_encoding, attr, OperationBasedEncoding(instance))
    return getattr(demo_encoding, attr)


def demo_solution(optimum: bool = False) -> Gantt:
    """
    Create a demo solution.

    :param optimum: should we return the optimal solution?
    :return: the demo solution
    """
    space: Final[GanttSpace] = demo_solution_space()
    result: Final[Gantt] = space.create()
    demo_encoding().map(demo_point_in_search_space(optimum=optimum),
                        result)
    space.validate(result)
    return result


def __make_gantt_demo_name(optimum: bool,
                           with_makespan: bool,
                           with_lower_bound: bool) -> str:
    """
    Construct the name for the demo gantt chart.

    :param optimum: should we return the optimal solution?
    :param with_makespan: should the makespan be included?
    :param with_lower_bound: should the lower bound be included?
    :return: the file name
    """
    prefix: str = "gantt_demo_opt_" if optimum else "gantt_demo_"
    if with_makespan:
        if with_lower_bound:
            return prefix + "with_makespan_and_lb"
        return prefix + "with_makespan"
    if with_lower_bound:
        return prefix + "with_lb"
    return prefix + "without_makespan"


def demo_gantt_chart(dirname: str,
                     optimum: bool = False,
                     with_makespan: bool = False,
                     with_lower_bound: bool = False,
                     width: Union[float, int, None] = cm_to_inch(10),
                     height: Union[float, int, None] = None,
                     filename: Union[str, Callable] =
                     __make_gantt_demo_name) -> List[Path]:
    """
    Plot the demo gantt chart.

    :param dirname: the directory
    :param optimum: should we return the optimal solution?
    :param with_makespan: should the makespan be included?
    :param with_lower_bound: should the lower bound be included?
    :param width: the optional width
    :param height: the optional height
    :param filename: the file name
    """
    the_dir: Final[Path] = Path.path(dirname)
    the_dir.ensure_dir_exists()

    if callable(filename):
        filename = filename(optimum,
                            with_makespan,
                            with_lower_bound)
    if not isinstance(filename, str):
        raise type_error(filename, "filename", str)

    result: Final[List[Path]] = []
    markers: List[Callable] = []

    if with_makespan:
        def info(g: Gantt):
            return Lang.current().format("gantt_info", gantt=g)
        if not optimum:
            markers.append(marker_makespan)
    else:
        def info(g: Gantt):
            return Lang.current().format("gantt_info_no_ms", gantt=g)

    if with_lower_bound:
        markers.append(marker_lb)

    for lang in Lang.all():
        lang.set_current()
        figure: Figure = create_figure(width=width, height=height)

        gantt = demo_solution(optimum=optimum)
        plot_gantt_chart(gantt=gantt,
                         figure=figure,
                         x_label_inside=False,
                         y_label_inside=False,
                         markers=markers,
                         info=info)

        result.extend(save_figure(fig=figure,
                                  dir_name=the_dir,
                                  file_name=lang.filename(filename),
                                  formats="svg"))
    return result


def makespan_lower_bound_table(
        dirname: str,
        filename: str = "makespan_lower_bound",
        instances: Iterable[str] =
        tuple(["demo"] + list(EXPERIMENT_INSTANCES))) -> Path:
    """
    Make a table with the makespan lower bounds.

    Larger lower bounds are taken from the repository
    https://github.com/thomasWeise/jsspInstancesAndResults.

    :param dirname: the directory where to store the generated file.
    :param filename: the filename
    :param instances: the instances
    :returns: the generated file
    """
    insts = list(instances)
    insts.sort()
    i = insts.index("demo")
    if i >= 0:
        del insts[i]
        insts.insert(0, "demo")

    # get the data with the lower bounds information
    url: Final[str] = "https://github.com/thomasWeise/jsspInstancesAndResults"
    logger(f"now loading data from '{url}'.")
    with rq.urlopen(url) as f:  # nosec
        data: str = f.read().decode(UTF8).strip()
    if not data:
        raise ValueError(f"Could not load data form {url}.")
    logger(f"finished loading data from '{url}'.")
    start = data.find("<table>")
    if start <= 0:
        raise ValueError(f"Could not find <table> in {data}.")
    end = data.rfind("</dl>", start)
    if end <= start:
        raise ValueError(f"Could not find </dl> in {data}.")
    data = data[start:end].strip()

    lang: Final[Lang] = Lang.current()

    text: Final[List[str]] = [
        r"|name|$\jsspJobs$|$\jsspMachines$|$\lowerBound(\objf)$|"
        + r"$\lowerBound(\objf)^{\star}$|source for&nbsp;"
        + r"$\lowerBound(\objf)^{\star}$|",
        "|:--|--:|--:|--:|--:|:--|"
    ]
    bsrc: Final[str] = "[@eq:jsspLowerBound]"

    # compute the data
    for instn in insts:
        solved_to_optimality: bool = False
        inst = Instance.from_resource(instn)
        lb = compute_makespan_lower_bound(machines=inst.machines,
                                          jobs=inst.jobs,
                                          matrix=inst)
        src = bsrc
        if instn == "demo":
            lbx = lb
            solved_to_optimality = True
        else:
            start = data.find(
                f'<td align="right"><strong>{instn}</strong></td>')
            if start <= 0:
                start = data.find(f'<td align="right">{instn}</td>')
                if start <= 0:
                    raise ValueError(f"Could not find instance {instn}.")
                prefix = ""
                suffix = "</td>"
                solved_to_optimality = True
            else:
                prefix = "<strong>"
                suffix = "</strong></td>"
            end = data.find("</tr>", start)
            if end <= start:
                raise ValueError(
                    f"Could not find end of instance {instn}.")
            sel = data[start:end].strip()
            logger(f"located instance text '{sel}' for {instn}.")
            prefix = f'<td align="right">{inst.jobs}</td>\n<td align="' \
                     f'right">{inst.machines}</td>\n<td align="right">' \
                     f'{prefix}'
            pi = sel.find(prefix)
            if pi <= 0:
                raise ValueError(
                    f"Could not find prefix for instance {instn}.")
            sel = sel[(pi + len(prefix)):]
            si = sel.find(suffix)
            if si <= 0:
                raise ValueError(
                    f"Could not find suffix for instance {instn}.")
            lbx = int(sel[0:si])
            npt = '<td align="center"><a href="#'
            npi = sel.find(npt, si)
            if npi < si:
                raise ValueError(
                    f"Could not find source start for instance {instn}.")
            sel = sel[npi + len(npt):]
            end = sel.find('"')
            if end <= 0:
                raise ValueError(
                    f"Could not find source end for instance {instn}.")
            srcname = sel[:end].strip()
            logger(f"Source name is {srcname}.")

            fsrc = data.find(f'<dt id="user-content-{srcname.lower()}">'
                             f'{srcname}</dt><dd>')
            if fsrc <= 0:
                raise ValueError(
                    f"Could not find source mark for instance {instn}.")
            fsrce = data.find("</dd>", fsrc)
            if fsrce <= fsrc:
                raise ValueError(
                    f"Could not find end mark for instance {instn}.")
            sel = data[fsrc:fsrce].strip()
            nm = sel.rfind("</a>")
            if nm <= 0:
                raise ValueError(
                    f"Could not find link end for instance {instn}"
                    f" and source name {srcname}.")
            fm = sel.rfind(">", 0, nm)
            if fm >= nm:
                raise ValueError(
                    f"Could not find label mark for instance {instn}"
                    f" and source name {srcname}.")
            src = f"[@{sel[(fm + 1):nm].strip()}]"

        if lbx <= lb:
            if lbx < lb:
                raise ValueError(f"Lbx {lbx} cannot be larger than lb {lb}.")
            if not solved_to_optimality:
                src = "[@eq:jsspLowerBound]"
        instn = f"`{instn}`"
        if not solved_to_optimality:
            instn = f"**{instn}**"

        text.append(
            f"|{instn}|{inst.machines}|{inst.jobs}|{lb}|{lbx}|{src}|")

    # write the result
    out_dir = Path.path(dirname)
    out_dir.ensure_dir_exists()
    out_file = out_dir.resolve_inside(lang.filename(filename) + ".md")
    out_file.write_all(text)
    out_file.enforce_file()
    logger(f"finished writing output results to file '{out_file}'.")
    return out_file


# are we being executed?
if __name__ == '__main__':
    dest_dir: Final[Path] = Path.path(sys.argv[1])
    dest_dir.ensure_dir_exists()
    logger(f"We will print the JSSP examples into dir '{dest_dir}'.")
    makespan_lower_bound_table(dest_dir)
    demo_gantt_chart(dest_dir, True)
    demo_gantt_chart(dest_dir, False)
    logger(f"Finished printing the JSSP examples into dir '{dest_dir}'.")
