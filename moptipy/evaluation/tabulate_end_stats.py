"""Make an LaTeX end-statistics table with column wrapping."""

from io import TextIOBase
from math import inf
from typing import Any, Callable, Final, Iterable, TextIO

from pycommons.io.console import logger
from pycommons.io.path import line_writer

from moptipy.evaluation.end_statistics import EndStatistics
from moptipy.evaluation.end_statistics import getter as es_getter
from moptipy.utils.number_renderer import default_get_int_renderer

#: the integer number renderer
__INT_2_STR: Final[Callable[[int], str]] = default_get_int_renderer()


def tabulate_end_stats(
        data: Iterable[EndStatistics],
        dest: Callable[[int], TextIO | TextIOBase],
        n_wrap: int = 3,
        max_rows: int = 50,
        stats: Iterable[tuple[Callable[[
            EndStatistics], int | float | None], str, bool,
        Callable[[int | float], str]]] = ((
        es_getter("bestF.mean"),
        r"\bestFmean", True,
        lambda v: __INT_2_STR(round(v))), ),
        instance_get: Callable[[EndStatistics], str] =
        lambda es: es.instance,
        instance_sort_key: Callable[[str], Any] = lambda x: x,
        instance_name: Callable[[str], str] =
        lambda x: f"{{\\instStyle{{{x}}}}}",
        algorithm_get: Callable[[EndStatistics], str] = lambda x: x.algorithm,
        algorithm_sort_key: Callable[[str], Any] = lambda x: x,
        algorithm_name: Callable[[str], str] = lambda x: x,
        instance_cols: Iterable[tuple[str, Callable[[str], str]]] = (),
        best_format: Callable[[str], str] =
        lambda x: f"{{\\textbf{{{x}}}}}",
        instance_header: str = "instance",
        best_count_header: str | None = r"\nBest") -> None:
    """
    Make a table of end statistics that can wrap multiple pages, if need be.

    :param data: the source data
    :param dest: the destination generator
    :param n_wrap: the number of times we can wrap a table
    :param max_rows: the maximum rows per destination
    :param stats: the set of statistics: tuples of statistic, title, whether
        minimization or maximization, and a to-string converter
    :param instance_get: get the instance identifier
    :param instance_name: get the instance name, as it should be printed
    :param instance_sort_key: get the sort key for the instance
    :param algorithm_get: get the algorithm identifier
    :param algorithm_name: get the algorithm name, as it should be printed
    :param algorithm_sort_key: get the sort key for the algorithm
    :param instance_cols: the fixed instance columns
    :param best_format: format the best value
    :param instance_header: the header for the instance
    :param best_count_header: the header for the best count
    """
    if not isinstance(data, list):
        data = list(data)
    if list.__len__(data) <= 0:
        raise ValueError("Empty data?")
    if not isinstance(stats, list):
        stats = list(stats)
    n_stats: Final[int] = list.__len__(stats)
    if n_stats <= 0:
        raise ValueError("No statistics data?")

    # We create a map of data
    datamap: Final[dict[str, dict[str, EndStatistics]]] = {}
    for es in data:
        algorithm: str = algorithm_get(es)
        instance: str = instance_get(es)
        if instance in datamap:
            datamap[instance][algorithm] = es
        else:
            datamap[instance] = {algorithm: es}

    instances: Final[list[str]] = sorted(
        {es.instance for es in data}, key=instance_sort_key)
    if list.__len__(instances) <= 0:
        raise ValueError("No instance!")

    algorithms: Final[list[str]] = sorted({
        k for x in datamap.values() for k in x}, key=algorithm_sort_key)
    n_algorithms: Final[int] = list.__len__(algorithms)
    if n_algorithms <= 0:
        raise ValueError("No algorithms!")
    n_data_cols: Final[int] = n_algorithms * n_stats

    if not isinstance(instance_cols, tuple):
        instance_cols = tuple(instance_cols)
    n_inst_cols: Final[int] = tuple.__len__(instance_cols)

    # Now we compute all the values
    output: list[list[str]] = []

    best_count: Final[list[int]] = [0] * n_data_cols
    for instance in instances:
        row: list[str] = [instance_name(instance)]
        row.extend(ic[1](instance) for ic in instance_cols)
        values: list[int | float | None] = [None] * n_data_cols
        best: list[int | float] = [
            (inf if stat[2] else -inf) for stat in stats]

        # get the values
        idx: int = 0
        for algorithm in algorithms:
            es = datamap[instance][algorithm]
            for si, stat in enumerate(stats):
                values[idx] = value = stat[0](es)
                idx += 1
                if (value is not None) and ((stat[2] and (
                        value < best[si])) or ((not stat[2]) and (
                        value > best[si]))):
                    best[si] = value

        # format the values
        idx = 0
        for _ in range(n_algorithms):
            for si, stat in enumerate(stats):
                value = values[idx]
                idx += 1
                if value is None:
                    row.append("")
                    continue
                printer: str = stat[3](value)
                if ((stat[2] and (value <= best[si])) or ((not stat[2]) and (
                        value >= best[si]))):
                    best_count[idx - 1] += 1
                    printer = best_format(printer)
                row.append(printer)
        output.append(row)

    if best_count_header is not None:
        nc: int = n_inst_cols + 1
        row = [f"\\multicolumn{{{nc}}}{{r}}{{{best_count_header}}}"]
        best = [-1] * n_stats

        # get the best best counts
        idx = 0
        for _ in range(n_algorithms):
            for si in range(n_stats):
                value = best_count[idx]
                best[si] = max(value, best[si])
                idx += 1

        # format the best count values
        idx = 0
        for _ in range(n_algorithms):
            for si in range(n_stats):
                ivalue: int = best_count[idx]
                idx += 1
                printer = str(ivalue)
                if ivalue >= best[si]:
                    printer = best_format(printer)
                row.append(printer)
        output.append(row)

    # now we got all the data prepared and just need to arrange it
    n_output: Final[int] = list.__len__(output)
    pages: Final[list[list[list[int]]]] = []
    idx = 0
    keep_going: bool = True
    while keep_going:
        current_page: list[list[int]] = []
        pages.append(current_page)
        for _ in range(max_rows):
            current_row: list[int] = []
            current_page.append(current_row)
            for _ in range(n_wrap):
                current_row.append(idx)
                idx += 1
                if idx >= n_output:
                    keep_going = False
                    break
            if not keep_going:
                break
    # ok, now we know the data that we can put on each page
    # we now re-arrange it
    for current_page in pages:
        ids: list[int] = sorted(x for y in current_page for x in y)
        idx = 0
        for col in range(n_wrap):
            for current_row in current_page:
                if col < list.__len__(current_row):
                    current_row[col] = ids[idx]
                    idx += 1

    # now we construct the header and footer that should go into each
    # partial table
    header: list[str] = []
    footer: list[str] = [r"\hline%", r"\end{tabular}%"]

    writer: list[str] = ["l"]
    writer.extend("r" for _ in range(n_inst_cols))
    writer.extend(["r"] * n_stats * n_algorithms)
    txt: str = "".join(writer)
    header.extend((f"\\begin{{tabular}}{{{'|'.join([txt] * n_wrap)}}}%",
                   r"\hline%"))

    writer.clear()
    writer.append(instance_header)
    writer.extend(f"\\multicolumn{{1}}{{c}}{{{ic[0]}}}"
                  for ic in instance_cols)
    writer.extend(f"\\multicolumn{{{n_stats}}}{{c}}{{{algorithm_name(algo)}}}"
                  for algo in algorithms)
    txt = "&".join(writer)
    if n_wrap > 1:
        writer[-1] = writer[-1].replace("{c}", "{c|}")
        txt_inner = "&".join(writer)
        header.append(f"{'&'.join([txt_inner] * (n_wrap - 1))}&{txt}\\\\%")
    else:
        header.append(f"{txt}\\\\%")

    writer.clear()
    if n_stats > 1:
        writer.clear()
        writer.append("")
        writer.extend("" for _ in range(n_inst_cols))
        writer.extend(f"\\multicolumn{{1}}{{c}}{{{sss[2]}}}"
                      for _ in range(n_algorithms) for sss in stats)
        txt = "&".join(writer)
        if n_wrap > 1:
            writer[-1] = writer[-1].replace("{c}", "{c|}")
            txt_inner = "&".join(writer)
            header.append(f"{'&'.join([txt_inner] * (n_wrap - 1))}&{txt}\\\\%")
        else:
            header.append(f"{txt}\\\\%")
        writer.clear()
    header.append(r"\hline%")

    for page_idx, current_page in enumerate(pages):
        logger(f"Now tacking data part {page_idx + 1}.")
        with dest(page_idx + 1) as stream:
            out_stream = line_writer(stream)
            for txt in header:
                out_stream(txt)
            for current_row in current_page:
                line: list[str] = [x for y in current_row for x in output[y]]
                out_stream(f"{'&'.join(line)}\\\\%")
            for txt in footer:
                out_stream(txt)
