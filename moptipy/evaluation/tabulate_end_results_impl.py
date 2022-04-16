"""A method to tabulate end results."""

from math import inf, isfinite, nan, isnan
from typing import Callable, Final, Iterable, Optional, List, Union, cast, \
    Dict, Any

from moptipy.api.logging import KEY_ALGORITHM, KEY_INSTANCE, KEY_GOAL_F
from moptipy.api.logging import KEY_BEST_F, KEY_TOTAL_FES, \
    KEY_LAST_IMPROVEMENT_TIME_MILLIS, KEY_TOTAL_TIME_MILLIS, \
    KEY_LAST_IMPROVEMENT_FE
from moptipy.evaluation.base import F_NAME_RAW, F_NAME_SCALED
from moptipy.evaluation.end_results import EndResult, KEY_FES_PER_MS
from moptipy.evaluation.end_statistics import EndStatistics, KEY_BEST_F_SCALED
from moptipy.evaluation.statistics import KEY_MINIMUM, KEY_MEAN_ARITH, \
    KEY_STDDEV, KEY_MEAN_GEOM, KEY_MEDIAN, KEY_MAXIMUM
from moptipy.utils.lang import Lang
from moptipy.utils.logger import SCOPE_SEPARATOR
from moptipy.utils.markdown import Markdown
from moptipy.utils.path import Path
from moptipy.utils.strings import numbers_to_strings
from moptipy.utils.table import Table
from moptipy.utils.text_format import TextFormatDriver, FormattedStr
from moptipy.utils.types import type_error


def default_column_namer(col: str) -> str:
    """
    Get the default name for columns.

    :param col: the column identifier
    :returns: the column name
    """
    if not isinstance(col, str):
        raise type_error(col, "column name", str)
    if col == KEY_INSTANCE:
        return "$\\instance$"
    if col == KEY_ALGORITHM:
        return Lang.translate("setup")
    if col in ("lower_bound", "lower_bound_short", KEY_GOAL_F):
        return "$\\lowerBound{\\objf}$"
    if col == "summary":
        return Lang.translate(col)

    if SCOPE_SEPARATOR not in col:
        raise ValueError(
            f"statistic '{col}' should contain '{SCOPE_SEPARATOR}'.")
    key, stat = col.split(SCOPE_SEPARATOR)
    if (len(key) <= 0) or (len(stat) <= 0):
        raise ValueError(f"invalid statistic '{col}'.")

    if key == KEY_BEST_F_SCALED:
        key = F_NAME_SCALED
    elif key == KEY_BEST_F:
        key = F_NAME_RAW

    # now fix statistics part
    if stat == KEY_MEAN_ARITH:
        stat = "mean"
    elif stat == KEY_STDDEV:
        stat = "sd"
    elif stat == KEY_MEAN_GEOM:
        stat = "gmean"
    elif stat == KEY_MEDIAN:
        stat = "med"
    elif stat == KEY_MINIMUM:
        if key in (F_NAME_RAW, F_NAME_SCALED):
            stat = Lang.translate("best")
        else:
            stat = "min"
    elif stat == KEY_MAXIMUM:
        if key in (F_NAME_RAW, F_NAME_SCALED):
            stat = Lang.translate("worst")
        else:
            stat = "max"
    else:
        raise ValueError(f"unknown statistic '{stat}' for '{col}'.")

    if key == F_NAME_RAW:
        return stat
    if key == F_NAME_SCALED:
        return f"{stat}1"
    if key == KEY_TOTAL_FES:
        key = "FEs"
    elif key == KEY_LAST_IMPROVEMENT_TIME_MILLIS:
        key = "t"
    elif key == KEY_TOTAL_TIME_MILLIS:
        key = "T"
    elif key == KEY_FES_PER_MS:
        key = "FE/ms"
    return f"{stat}({key})"


def __finite_max(data: Iterable[Union[int, float, None]]) \
        -> Union[int, float]:
    """
    Compute the finite maximum of a data column.

    :param data: the data to iterate over
    :returns: the finite maximum, or `nan` if none can be found or if there
        is only a single value
    """
    if not isinstance(data, Iterable):
        raise type_error(data, "data", Iterable)
    maxi: Union[int, float] = -inf
    count: int = 0
    for d in data:
        count += 1
        if d is None:
            continue
        if isfinite(d) and (d > maxi):
            maxi = d
    return maxi if (count > 1) and isfinite(maxi) else nan


def __finite_min(data: Iterable[Union[int, float, None]]) \
        -> Union[int, float]:
    """
    Compute the finite minimum of a data column.

    :param data: the data to iterate over
    :returns: the finite minimum, or `nan` if none can be found
    """
    if not isinstance(data, Iterable):
        raise type_error(data, "data", Iterable)
    mini: Union[int, float] = inf
    count: int = 0
    for d in data:
        count += 1
        if d is None:
            continue
        if isfinite(d) and (d < mini):
            mini = d
    return mini if (count > 1) and isfinite(mini) else nan


def __nan(data: Iterable[Union[int, float, None]]) -> float:
    """
    Get `nan`.

    :param data: ignored
    """
    if not isinstance(data, Iterable):
        raise type_error(data, "data", Iterable)
    return nan


def default_column_best(col: str) ->\
        Callable[[Iterable[Union[int, float, None]]], Union[int, float]]:
    """
    Get a function to compute the best value in a column.

    The returned function can compute the best value in a column. If no value
    is best, it should return `nan`.

    :param col: the column name string
    :returns: a function that can compute the best value per column
    """
    if not isinstance(col, str):
        raise type_error(col, "column name", str)

    if col in (KEY_INSTANCE, KEY_ALGORITHM, "lower_bound",
               "lower_bound_short", KEY_GOAL_F):
        return __nan

    if SCOPE_SEPARATOR not in col:
        raise ValueError(
            f"statistic '{col}' should contain '{SCOPE_SEPARATOR}'.")
    key, stat = col.split(SCOPE_SEPARATOR)
    if (len(key) <= 0) or (len(stat) <= 0):
        raise ValueError(f"invalid statistic '{col}'.")

    if stat == "sd":
        return __finite_min

    if key in (KEY_BEST_F_SCALED, F_NAME_SCALED, KEY_BEST_F, F_NAME_RAW):
        return __finite_min
    if key in (KEY_FES_PER_MS, KEY_LAST_IMPROVEMENT_TIME_MILLIS,
               KEY_LAST_IMPROVEMENT_FE, KEY_TOTAL_FES, KEY_TOTAL_TIME_MILLIS):
        return __finite_max

    return __nan


def __getter(s: str) -> Callable[[EndStatistics], Union[int, float, None]]:
    """
    Obtain a getter for the end statistics.

    :param s: the name
    :returns: the getter
    """
    if not isinstance(s, str):
        raise type_error(s, "getter name", str)
    getter = EndStatistics.getter(s)

    def __fixed(e: EndStatistics, g=getter, n=s) -> Union[int, float, None]:
        res = g(e)
        if res is None:
            return None
        if not isinstance(res, (int, float)):
            raise type_error(res, f"result of getter '{n}' for statistic {e}",
                             (int, float))
        return res
    return __fixed


def tabulate_end_results(
        end_results: Iterable[EndResult],
        file_name: str = "table",
        dir_name: str = ".",
        algorithm_instance_statistics: Iterable[str] = (
            f"{KEY_BEST_F}{SCOPE_SEPARATOR}{KEY_MINIMUM}",
            f"{KEY_BEST_F}{SCOPE_SEPARATOR}{KEY_MEAN_ARITH}",
            f"{KEY_BEST_F}{SCOPE_SEPARATOR}{KEY_STDDEV}",
            f"{KEY_BEST_F_SCALED}{SCOPE_SEPARATOR}{KEY_MEAN_ARITH}",
            f"{KEY_FES_PER_MS}{SCOPE_SEPARATOR}{KEY_MEAN_ARITH}",
            f"{KEY_LAST_IMPROVEMENT_TIME_MILLIS}{SCOPE_SEPARATOR}"
            f"{KEY_MEAN_ARITH}",),
        algorithm_summary_statistics: Optional[Iterable[Optional[str]]] = (
            f"{KEY_BEST_F_SCALED}{SCOPE_SEPARATOR}{KEY_MINIMUM}",
            f"{KEY_BEST_F_SCALED}{SCOPE_SEPARATOR}{KEY_MEAN_GEOM}",
            f"{KEY_BEST_F_SCALED}{SCOPE_SEPARATOR}{KEY_MAXIMUM}",
            f"{KEY_BEST_F_SCALED}{SCOPE_SEPARATOR}{KEY_STDDEV}",
            f"{KEY_FES_PER_MS}{SCOPE_SEPARATOR}{KEY_MEAN_GEOM}",
            f"{KEY_LAST_IMPROVEMENT_TIME_MILLIS}{SCOPE_SEPARATOR}"
            f"{KEY_MEAN_GEOM}"
        ),
        text_format_driver: Union[TextFormatDriver,
                                  Callable[[], TextFormatDriver]]
        = Markdown.instance,
        algorithm_sorter: Callable[[str], Any] = lambda a: a,
        instance_sorter: Callable[[str], Any] = lambda i: i,
        col_namer: Callable[[str], str] = default_column_namer,
        col_best: Callable[[str], Callable[[Iterable[Union[
            int, float, None]]], Union[int, float]]] = default_column_best,
        put_lower_bound: bool = True,
        lower_bound_getter: Optional[Callable[[EndStatistics],
                                              Union[int, float, None]]] =
        __getter(KEY_GOAL_F),
        lower_bound_name: Optional[str] = "lower_bound") -> Path:
    r"""
    Tabulate the statistics about the end results of an experiment.

    A two-part table is produced. In the first part, it presents summary
    statistics about each instance-algorithm combination, sorted by instance.
    In the second part, it presents summary statistics of the algorithms over
    all instances. The following default columns are provided:

    1. Part 1: Algorithm-Instance statistics
        - `$\instance$`: the instance name
        - `$\lowerBound{\objf}$`: the lower bound of the objective value of
          the instance
        - `setup`: the name of the algorithm or algorithm setup
        - `best`: the best objective value reached by any run on that instance
        - `mean`: the arithmetic mean of the best objective values reached
          over all runs
        - `sd`: the standard deviation of the best objective values reached
          over all runs
        - `mean1`: the arithmetic mean of the best objective values reached
          over all runs, divided by the lower bound (or goal objective value)
        - `mean(FE/ms)`: the arithmetic mean of objective function evaluations
           performed per millisecond, over all runs
        - `mean(t)`: the arithmetic mean of the time in milliseconds when the
          last improving move of a run was applied, over all runs

    2. Part 2: Algorithm Summary Statistics
        - `setup`: the name of the algorithm or algorithm setup
        - `best1`: the minimum of the best objective values reached divided by
          the lower bound (or goal objective value) over all runs
        - `gmean1`: the geometric mean of the best objective values reached
           divided by the lower bound (or goal objective value) over all runs
        - `worst1`: the maximum of the best objective values reached divided
          by the lower bound (or goal objective value) over all runs
        - `sd1`: the standard deviation of the best objective values reached
           divided by the lower bound (or goal objective value) over all runs
        - `gmean(FE/ms)`: the geometric mean of objective function evaluations
          performed per millisecond, over all runs
        - `gmean(t)`: the geometric mean of the time in milliseconds when the
          last improving move of a run was applied, over all runs

    You can freely configure which columns you want for each part and whether
    you want to have the second part included. Also, for each group of values,
    the best one is marked in bold face.

    :param end_results: the end results data
    :param file_name: the base file name
    :param dir_name: the base directory
    :param algorithm_instance_statistics: the statistics to print
    :param algorithm_summary_statistics: the summary statistics to print per
        algorithm
    :param text_format_driver: the text format driver
    :param algorithm_sorter: a function returning sort keys for algorithms
    :param instance_sorter: a function returning sort keys for instances
    :param col_namer: the column namer function
    :param col_best: the column-best getter function
    :param put_lower_bound: should we put the lower bound or goal objective
        value?
    :param lower_bound_getter: the getter for the lower bound
    :param lower_bound_name: the name key for the lower bound to be passed
        to `col_namer`
    :returns: the path to the file with the tabulated end results
    """
    # initial type checks
    if not isinstance(end_results, Iterable):
        raise type_error(end_results, "end_results", Iterable)
    if not isinstance(file_name, str):
        raise type_error(file_name, "file_name", str)
    if not isinstance(dir_name, str):
        raise type_error(dir_name, "dir_name", str)
    if not isinstance(algorithm_instance_statistics, Iterable):
        raise type_error(algorithm_instance_statistics,
                         "algorithm_instance_statistics", Iterable)
    if (algorithm_summary_statistics is not None)\
            and (not isinstance(algorithm_instance_statistics, Iterable)):
        raise type_error(algorithm_summary_statistics,
                         "algorithm_summary_statistics", (Iterable, None))
    if not isinstance(put_lower_bound, bool):
        raise type_error(put_lower_bound, "put_lower_bound", bool)
    if put_lower_bound:
        if not callable(lower_bound_getter):
            raise type_error(lower_bound_getter, "lower_bound_getter",
                             call=True)
        if not isinstance(lower_bound_name, str):
            raise type_error(lower_bound_name, "lower_bound_name", str)
    if callable(text_format_driver):
        text_format_driver = text_format_driver()
    if not isinstance(text_format_driver, TextFormatDriver):
        raise type_error(text_format_driver, "text_format_driver",
                         TextFormatDriver, True)
    exponent_renderer: Final[Callable[[str], str]] = \
        text_format_driver.render_numeric_exponent
    if not callable(col_namer):
        raise type_error(col_namer, "col_namer", call=True)
    if not callable(col_best):
        raise type_error(col_best, "col_best", call=True)

    # get the getters
    algo_inst_getters: Final[List[Callable[[EndStatistics],
                                           Union[int, float, None]]]] = \
        [__getter(d) for d in algorithm_instance_statistics]
    n_algo_inst_getters: Final[int] = len(algo_inst_getters)
    if n_algo_inst_getters <= 0:
        raise ValueError("algorithm-instance dimensions must not be empty.")
    algo_getters: Final[Optional[List[Optional[
        Callable[[EndStatistics], Union[int, float, None]]]]]] = \
        (None if (algorithm_summary_statistics is None)
         else [None if (d is None) else __getter(d)
               for d in cast(Iterable, algorithm_summary_statistics)])
    if algo_getters is not None:
        if len(algo_getters) != n_algo_inst_getters:
            raise ValueError(
                f"there are {n_algo_inst_getters} algorithm-instance columns,"
                f" but {len(algo_getters)} algorithms summary columns.")
        if all(g is None for g in algo_getters):
            raise ValueError(
                "if all elements of algorithm_summary_statistics are None, "
                "then specify algorithm_summary_statistics=None")

    # gather the statistics for each algorithm-instance combination
    algo_inst_list: Final[List[EndStatistics]] = []
    EndStatistics.from_end_results(end_results, algo_inst_list.append)
    if len(algo_inst_list) <= 0:
        raise ValueError("no algorithm-instance combinations?")
    # get the sorted lists of algorithms and instances
    insts: Final[List[str]] = sorted({s.instance for s in algo_inst_list},
                                     key=instance_sorter)
    n_insts: Final[int] = len(insts)
    if n_insts <= 0:
        raise ValueError("no instance found?")
    algos: Final[List[str]] = sorted({s.algorithm for s in algo_inst_list},
                                     key=algorithm_sorter)
    n_algos: Final[int] = len(algos)
    if n_algos <= 0:
        raise ValueError("no algos found?")

    # finalize the data dictionaries: d[inst][algo] = stats
    algo_inst_dict: Final[Dict[str, Dict[str, EndStatistics]]] = {}
    for e in algo_inst_list:
        if e.instance not in algo_inst_dict:
            algo_inst_dict[e.instance] = {}
        algo_inst_dict[e.instance][e.algorithm] = e
    for ina in insts:
        if len(algo_inst_dict[ina]) != n_algos:
            raise ValueError(
                f"expected {n_algos} entries for instance '{ina}', but "
                f"got only {len(algo_inst_dict[ina])}, namely "
                f"{algo_inst_dict[ina].keys()} instead of {algos}.")

    # compute the per-instance lower bounds if we need them
    lower_bounds: Optional[List[str]]
    if put_lower_bound:
        lb: List[Union[int, float, None]] = []
        for inst in insts:
            bounds = list({lower_bound_getter(d)
                           for d in algo_inst_dict[inst].values()})
            if len(bounds) != 1:
                raise ValueError(f"inconsistent lower bounds {bounds} for "
                                 f"instance '{inst}'.")
            lb.append(bounds[0])
        lower_bounds = numbers_to_strings(lb,
                                          exponent_renderer=exponent_renderer)
        del lb
    else:
        lower_bounds = None
    del algo_inst_list

    # gather the algorithm summary statistics
    algo_dict: Final[Optional[Dict[str, EndStatistics]]] = {} \
        if (n_insts > 1) and (algo_getters is not None) else None
    if algo_dict is not None:
        def __put(es: EndStatistics):
            nonlocal algo_dict
            if es.algorithm in algo_dict:
                raise ValueError(f"already encountered {es.algorithm}?")
            algo_dict[es.algorithm] = es
        EndStatistics.from_end_results(end_results, __put,
                                       join_all_instances=True)
        del __put
    del end_results
    if len(algo_dict) != n_algos:
        raise ValueError(f"there are {n_algos} algorithms, but in the "
                         f"summary, only {len(algo_dict)} appear?")

    # set up column titles
    def __fix_name(s: str, nn=col_namer) -> str:
        if not isinstance(s, str):
            raise type_error(s, "column name", str)
        if len(s) <= 0:
            raise ValueError("string must not be empty!")
        na = nn(s)
        if not isinstance(na, str):
            raise type_error(na, f"name computed for {s}", str)
        if len(na) <= 0:
            raise ValueError(f"name computed for {s} cannot be empty.")
        return na

    algo_inst_cols: Final[List[str]] = \
        [__fix_name(s) for s in algorithm_instance_statistics]
    if len(algo_inst_cols) <= 0:
        raise ValueError("no algorithm_instance columns?")
    algo_cols: Optional[List[Optional[str]]] = \
        None if algo_dict is None else \
        [(None if s is None else __fix_name(s))
         for s in cast(Iterable, algorithm_summary_statistics)]
    if algo_cols == algo_inst_cols:
        algo_cols = None  # no need to repeat header if it is the same

    # set up the column definitions
    algo_inst_cols.insert(0, __fix_name(KEY_ALGORITHM))
    if put_lower_bound:
        algo_inst_cols.insert(0, __fix_name(lower_bound_name))
    algo_inst_cols.insert(0, __fix_name(KEY_INSTANCE))

    if algo_cols is not None:
        algo_cols.insert(0, __fix_name(KEY_ALGORITHM))
        if put_lower_bound:
            algo_cols.insert(0, None)
        algo_cols.insert(0, None)

    col_def: Final[str] = ("lrl" if put_lower_bound else "ll") \
        + ("r" * n_algo_inst_getters)

    # get the data columns of all columns and convert to strings
    # format: column -> columns data
    # we first need to get all the data at once to allow for a uniform
    # formatting via numbers_to_strings
    algo_inst_data_raw: Final[List[List[Union[int, float, None]]]] =\
        [[None if getter is None else getter(algo_inst_dict[inst][algo])
          for inst in insts for algo in algos]
         for getter in algo_inst_getters]
    algo_inst_strs_raw: Final[List[List[Optional[str]]]] = [
        numbers_to_strings(col, exponent_renderer=exponent_renderer)
        for col in algo_inst_data_raw]

    # now break the data into sections
    # format: column -> per-instance section -> section data
    # after we break the data in sections, we can mark the per-section bests
    # and we can flush the data to the table section-wise
    algo_inst_data: Final[List[List[List[Union[int, float, None]]]]] = \
        [[col[c * n_algos:(c + 1) * n_algos] for c in range(n_insts)]
         for col in algo_inst_data_raw]
    del algo_inst_data_raw
    algo_inst_strs: Final[List[List[List[Optional[str]]]]] = \
        [[col[c * n_algos:(c + 1) * n_algos] for c in range(n_insts)]
         for col in algo_inst_strs_raw]
    del algo_inst_strs_raw

    # now format the data, i.e., compute the per-section best value
    # of each column and mark it with bold face
    for col_idx, stat in enumerate(algorithm_instance_statistics):
        col_n = algo_inst_data[col_idx]
        col_s = algo_inst_strs[col_idx]
        best_getter = col_best(stat)
        if not callable(best_getter):
            raise type_error(best_getter, f"result of col_best for {stat}",
                             call=True)
        for chunk_idx, chunk_n in enumerate(col_n):
            chunk_s = col_s[chunk_idx]
            best = best_getter(chunk_n)
            if (best is not None) and (not isnan(best)):
                for idx, val in enumerate(chunk_n):
                    if val == best:
                        chunk_s[idx] = FormattedStr.add_format(
                            chunk_s[idx], bold=True)
    del algo_inst_data
    del algorithm_instance_statistics

    # now we pre-pend the instance and algorithm information
    algo_inst_strs.insert(0, [[FormattedStr.add_format(algo, code=True)
                               for algo in algos]] * n_insts)
    if put_lower_bound:
        algo_inst_strs.insert(0, [[b] for b in lower_bounds])
    algo_inst_strs.insert(0, [[
        FormattedStr.add_format(inst, code=True)] for inst in insts])
    del lower_bounds
    del insts

    algo_strs: Optional[List[List[Optional[str]]]] = None
    if (algo_dict is not None) and (algorithm_summary_statistics is not None):
        # get the data columns of the algorithm summaries
        # format: column -> columns data
        algo_data: Final[List[List[Union[int, float, None]]]] = \
            [[None if getter is None else getter(algo_dict[algo])
              for algo in algos]
             for getter in algo_getters]
        algo_strs = [numbers_to_strings(
            col, exponent_renderer=exponent_renderer) for col in algo_data]

        # now format the data, i.e., compute the per-section best value
        # of each column and mark it with bold face
        for col_idx, stat in enumerate(algorithm_summary_statistics):
            if stat is None:
                continue
            acol_n = algo_data[col_idx]
            acol_s = algo_strs[col_idx]
            best_getter = col_best(stat)
            if not callable(best_getter):
                raise type_error(
                    best_getter, f"result of col_best for {stat}", call=True)
            best = best_getter(acol_n)
            if (best is not None) and (not isnan(best)):
                for idx, val in enumerate(acol_n):
                    if val == best:
                        acol_s[idx] = FormattedStr.add_format(
                            acol_s[idx], bold=True)
        del algo_data
        algo_strs.insert(0, algos)
        if put_lower_bound:
            algo_strs.insert(0, [])
        algo_strs.insert(0, [__fix_name("summary")] * n_algos)

    # write the table
    dest: Final[Path] = text_format_driver.filename(file_name, dir_name)
    with dest.open_for_write() as wd:
        with Table(wd, col_def, text_format_driver) as table:
            table.header_row(algo_inst_cols)
            for i in range(n_algos):
                table.section_cols([col[i] for col in algo_inst_strs])
            if algo_strs is not None:
                table.section_cols(algo_strs, algo_cols)

    return dest