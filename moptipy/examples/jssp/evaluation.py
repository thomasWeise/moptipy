"""Evaluate the results of the example experiment."""
import argparse
import os.path as pp
from math import log2
from re import Match, Pattern
from re import compile as _compile
from statistics import median
from typing import Callable, Final, Iterable, cast

from pycommons.io.console import logger
from pycommons.io.path import Path, directory_path
from pycommons.strings.string_conv import num_to_str
from pycommons.types import type_error

from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import TIME_UNIT_FES, TIME_UNIT_MILLIS
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_results import from_csv as end_results_from_csv
from moptipy.evaluation.end_results import from_logs as end_results_from_logs
from moptipy.evaluation.end_results import to_csv as end_results_to_csv
from moptipy.evaluation.end_statistics import EndStatistics
from moptipy.evaluation.end_statistics import (
    from_end_results as es_from_end_results,
)
from moptipy.evaluation.end_statistics import to_csv as es_to_csv
from moptipy.evaluation.tabulate_end_results import (
    DEFAULT_ALGORITHM_INSTANCE_STATISTICS,
    DEFAULT_ALGORITHM_SUMMARY_STATISTICS,
    command_column_namer,
    tabulate_end_results,
)
from moptipy.evaluation.tabulate_result_tests import tabulate_result_tests
from moptipy.examples.jssp.experiment import ALGORITHMS, INSTANCES
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.plots import (
    plot_end_makespans,
    plot_end_makespans_over_param,
    plot_progresses,
    plot_stat_gantt_charts,
)
from moptipy.spaces.permutations import Permutations
from moptipy.utils.help import moptipy_argparser
from moptipy.utils.lang import EN
from moptipy.utils.logger import sanitize_name
from moptipy.utils.strings import (
    beautify_float_str,
    name_str_to_num,
)

#: The letter mu
LETTER_M: Final[str] = "\u03BC"
#: The letter lambda
LETTER_L: Final[str] = "\u03BB"

#: the pre-defined instance sort keys
__INST_SORT_KEYS: Final[dict[str, int]] = {
    __n: __i for __i, __n in enumerate(INSTANCES)
}

#: the name of the mu+1 EA
NAME_EA_MU_PLUS_1: Final[str] = f"{LETTER_M}+1_ea"
#: the name of the mu+mu EA
NAME_EA_MU_PLUS_MU: Final[str] = f"{LETTER_M}+{LETTER_M}_ea"
#: the name of the mu+ld(mu) EA
NAME_EA_MU_PLUS_LOG_MU: Final[str] = f"{LETTER_M}+log\u2082{LETTER_M}_ea"
#: the name of the mu+sqrt(mu) EA
NAME_EA_MU_PLUS_SQRT_MU: Final[str] = f"{LETTER_M}+\u221A{LETTER_M}_ea"


def __make_all_names() -> tuple[str, ...]:
    """
    Create an immutable list of all algorithm names.

    :returns: the tuple with all algorithm names
    """
    inst = Instance.from_resource("demo")
    space = Permutations.with_repetitions(inst.jobs, inst.machines)
    return tuple([str(a(inst, space)) for a in ALGORITHMS])


#: the list of all algorithm names from the experiment
ALL_NAMES: Final[tuple[str, ...]] = __make_all_names()
del __make_all_names


def ea_family(name: str) -> str:
    """
    Name the evolutionary algorithm setup.

    :param name: the full name
    :returns: the short name of the family
    """
    if name.startswith("ea_"):
        ss: Final[list[str]] = name.split("_")
        ll: Final[int] = len(ss)
        if ll == 4:
            lambda_ = int(ss[2])
            if lambda_ == 1:
                return NAME_EA_MU_PLUS_1
            mu = int(ss[1])
            if lambda_ == int(log2(mu)):
                return NAME_EA_MU_PLUS_LOG_MU
            if lambda_ == round(mu ** 0.5):
                return NAME_EA_MU_PLUS_SQRT_MU
            ratio = lambda_ // mu
            if ratio * mu == lambda_:
                if ratio == 1:
                    return NAME_EA_MU_PLUS_MU
                return f"{LETTER_M}+{ratio}{LETTER_M}_ea"
        elif (ll == 6) and (ss[-2] == "gap") and (ss[-1] == "swap2"):
            return f"{ss[1]}+{ss[2]}_ea_br"
    raise ValueError(f"Invalid EA name {name!r}.")


def sa_family(name: str) -> str:
    """
    Name the simulated annealing setup.

    :param name: the full name
    :returns: the short name of the family
    """
    if name.startswith("sa_") and name.endswith("_swap2"):
        return (f"sa_T\u2080_"
                f"{beautify_float_str(name_str_to_num(name.split('_')[2]))}")
    raise ValueError(f"Invalid SA name {name!r}.")


def ma_family(name: str) -> str:
    """
    Compute the Memetic Algorithm family without the ls steps.

    :param name: the algorithm name
    :returns: the short name of the family
    """
    ss: Final[list[str]] = name.split("_")
    s = "rls" if name.startswith("marls") else "sa"
    return f"{ss[1]}+{ss[2]}_ma{s}"


def __make_algo_names() -> tuple[dict[str, int], dict[str, str]]:
    """
    Create the algorithm sort keys and name decode.

    :returns: the algorithm sort keys and name decode
    """
    names: list[str] = list(ALL_NAMES)
    names_new = list(ALL_NAMES)

    def __eacr(mtch: Match) -> str:
        """Transform the EA name with crossover."""
        br = name_str_to_num(mtch.group(3))
        if not 0.0 < br < 1.0:
            raise ValueError(f"{mtch} has invalid br={br}.")
        a = log2(br)
        b = int(a)
        c = log2(1.0 - br)
        d = int(c)
        q = f"(1-2^{d})" if abs(c - d) < abs(a - b) else f"2^{b}"
        q2 = num_to_str(br)
        if len(q2) <= (len(q) + 1):
            q = q2
        return f"{mtch.group(1)}+{mtch.group(2)}_ea_{q}"

    def __sa(mtch: Match) -> str:
        """Transform the SA name."""
        if mtch.group(0) == "sa_exp16_1em6_swap2":
            return "sa"
        return f"sa_{mtch.group(1)}_{mtch.group(2)}e-{mtch.group(3)}"

    def __ma(mtch: Match, suffix: str) -> str:
        """Transform the MA name."""
        if mtch.group(0) == "ma_2_2_1048576_gap_sa_exp16_5d1em6_swap2":
            return "2_masa_20"
        if mtch.group(0) == "marls_8_8_16_gap_swap2":
            return "8_marls_4"
        ls = name_str_to_num(mtch.group(3))
        if not 0 < ls < 1_000_000_000:
            raise ValueError(f"{mtch} has invalid ls={ls}.")
        e2 = int(log2(ls))
        if ls != (2 ** e2):
            raise ValueError(
                f"{mtch} has invalid ls={ls} since {e2}=log2({ls}).")
        e2s = f"2^{e2}"
        lss = str(ls)
        if len(lss) <= len(e2s):
            e2s = lss
        return f"{mtch.group(1)}+{mtch.group(2)}_ma{suffix}_{e2s}"

    # fix some names using regular expressions
    namer: dict[str, str] = {}
    used_names: set[str] = set(names)
    for pattern, repl in [("hc_swap2", "hc"), ("hc_swapn", "hcn"),
                          ("rls_swap2", "rls"), ("rls_swapn", "rlsn")]:
        re: Pattern = _compile(pattern)
        found = False
        for s in names:
            m: Match = re.match(s)
            if m is not None:
                ns = re.sub(cast(str | Callable[[Match], str], repl), s)
                if (ns is None) or (len(ns) <= 0):
                    raise ValueError(f"{s!r} -> {ns!r}?")
                if ns == s:
                    continue
                found = True
                os = namer.get(s)
                if os is not None:
                    if os == ns:
                        continue
                    raise ValueError(f"{s!r} -> {ns!r}, {os!r}?")
                if ns in used_names:
                    raise ValueError(f"Already got {ns!r}.")
                namer[s] = ns
                names_new.insert(names_new.index(s), ns)
        if not found:
            raise ValueError(f"did not find {pattern!r}.")

    return {__n: __i for __i, __n in enumerate(names_new)}, namer


#: the pre-defined algorithm sort keys and name map
__ALGO_SORT_KEYS, __ALGO_NAME_MAP = __make_algo_names()


def instance_sort_key(name: str) -> int:
    """
    Get the instance sort key for a given instance name.

    :param name: the instance name
    :returns: the sort key
    """
    if not isinstance(name, str):
        raise type_error(name, "name", str)
    if not name:
        raise ValueError("name must not be empty.")
    if name in __INST_SORT_KEYS:
        return __INST_SORT_KEYS[name]
    return 1000


def algorithm_sort_key(name: str) -> int:
    """
    Get the algorithm sort key for a given algorithm name.

    :param name: the algorithm name
    :returns: the sort key
    """
    if not isinstance(name, str):
        raise type_error(name, "name", str)
    if not name:
        raise ValueError("name must not be empty.")
    if name in __ALGO_SORT_KEYS:
        return __ALGO_SORT_KEYS[name]
    return 1000


def algorithm_namer(name: str) -> str:
    """
    Rename algorithm setups.

    :param name: the algorithm's original name
    :returns: the new name of the algorithm
    """
    if not isinstance(name, str):
        raise type_error(name, "name", str)
    if not name:
        raise ValueError("name must not be empty.")
    if name in __ALGO_NAME_MAP:
        return __ALGO_NAME_MAP[name]
    return name


def compute_end_results(results_dir: str,
                        dest_dir: str) -> Path:
    """
    Get the end results, compute them if necessary.

    :param results_dir: the results directory
    :param dest_dir: the destination directory
    :returns: the path to the end results file.
    """
    dest: Final[Path] = directory_path(dest_dir)
    results_file: Final[Path] = dest.resolve_inside("end_results.txt")
    if results_file.is_file():
        return results_file

    source: Final[Path] = directory_path(results_dir)
    logger(f"loading end results from path {source!r}.")
    results: Final[list[EndResult]] = list(end_results_from_logs(source))
    if not results:
        raise ValueError(f"Could not find any logs in {source!r}.")
    results.sort()
    logger(f"found {len(results)} log files in path {source!r}, storing "
           f"them in file {results_file!r}.")
    rf: Path = end_results_to_csv(results, results_file)
    if rf != results_file:
        raise ValueError(
            f"results file should be {results_file!r}, but is {rf!r}.")
    results_file.enforce_file()
    logger(f"finished writing file {results_file!r}.")
    return results_file


def get_end_results(
        file: str,
        insts: None | set[str] | Callable[[str], bool] = None,
        algos: None | set[str] | Callable[[str], bool] = None) \
        -> list[EndResult]:
    """
    Get a specific set of end results.

    :param file: the end results file
    :param insts: only these instances will be included if this parameter is
        provided
    :param algos: only these algorithms will be included if this parameter is
        provided
    """

    def __filter(er: EndResult,
                 ins=None if insts is None else
                 insts.__contains__ if isinstance(insts, set)
                 else insts,
                 alg=None if algos is None else
                 algos.__contains__ if isinstance(algos, set)
                 else algos) -> bool:
        if (ins is not None) and (not ins(er.instance)):
            return False
        return not ((alg is not None) and (not alg(er.algorithm)))

    col: Final[list[EndResult]] = list(end_results_from_csv(
        file=file, filterer=__filter))
    if len(col) <= 0:
        raise ValueError(
            f"no end results for instances {insts} and algorithms {algos}.")
    return col


def compute_end_statistics(end_results_file: str,
                           dest_dir: str) -> Path:
    """
    Get the end result statistics, compute them if necessary.

    :param end_results_file: the end results file
    :param dest_dir: the destination directory
    :returns: the path to the end result statistics file.
    """
    dest: Final[Path] = directory_path(dest_dir)
    stats_file: Final[Path] = dest.resolve_inside("end_statistics.txt")
    if stats_file.is_file():
        return stats_file

    results: Final[list[EndResult]] = get_end_results(end_results_file)
    if len(results) <= 0:
        raise ValueError("end results cannot be empty")
    stats: Final[list[EndStatistics]] = list(es_from_end_results(results))
    if len(stats) <= 0:
        raise ValueError("end result statistics cannot be empty")
    stats.sort()

    sf: Path = es_to_csv(stats, stats_file)
    if sf != stats_file:
        raise ValueError(f"stats file should be {stats_file!r} but is {sf!r}")
    stats_file.enforce_file()
    logger(f"finished writing file {stats_file!r}.")
    return stats_file


def table(end_results: Path, algos: list[str], dest: Path,
          swap_stats: Iterable[tuple[str, str]] | None = None) -> None:
    """
    Tabulate the end results.

    :param end_results: the path to the end results
    :param algos: the algorithms
    :param dest: the directory
    :param swap_stats: the statistics to swap out
    """
    EN.set_current()
    n: Final[str] = algorithm_namer(algos[0])
    algo_inst_stat: list | tuple = DEFAULT_ALGORITHM_INSTANCE_STATISTICS
    algo_sum_stat: list | tuple = DEFAULT_ALGORITHM_SUMMARY_STATISTICS
    if swap_stats is not None:
        algo_inst_stat = list(algo_inst_stat)
        algo_sum_stat = list(algo_sum_stat)
        for old, swap in swap_stats:
            found: bool = False
            for lst in [algo_inst_stat, algo_sum_stat]:
                for idx, elem in enumerate(lst):
                    if elem == old:
                        found = True
                        lst[idx] = swap
                        break
            if not found:
                raise ValueError(f"did not find {old!r} in {str(lst)!r}.")

    tabulate_end_results(
        end_results=get_end_results(end_results, algos=set(algos)),
        file_name=sanitize_name(f"end_results_{n}"), dir_name=dest,
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key,
        col_namer=command_column_namer,
        algorithm_namer=algorithm_namer,
        algorithm_instance_statistics=algo_inst_stat,
        algorithm_summary_statistics=algo_sum_stat,
        use_lang=False)


def tests(end_results: Path, algos: list[str], dest: Path) -> None:
    """
    Tabulate the end result tests.

    :param end_results: the path to the end results
    :param algos: the algorithms
    :param dest: the directory
    """
    EN.set_current()
    n: Final[str] = algorithm_namer(algos[0])
    tabulate_result_tests(
        end_results=get_end_results(end_results, algos=set(algos)),
        file_name=sanitize_name(f"tests_{n}"), dir_name=dest,
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key,
        algorithm_namer=algorithm_namer,
        use_lang=False)


def makespans(end_results: Path, algos: list[str], dest: Path,
              x_label_location: float = 1.0) -> None:
    """
    Plot the end makespans.

    :param end_results: the path to the end results
    :param algos: the algorithms
    :param dest: the directory
    :param x_label_location: the location of the label of the x-axis
    """
    n: Final[str] = algorithm_namer(algos[0])
    plot_end_makespans(
        end_results=get_end_results(end_results, algos=set(algos)),
        name_base=sanitize_name(f"makespan_scaled_{n}"), dest_dir=dest,
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key,
        algorithm_namer=algorithm_namer,
        x_label_location=x_label_location)


def gantt(end_results: Path, algo: str, dest: Path, source: Path,
          best: bool = False,
          insts: Iterable[str] | None = None) -> None:
    """
    Plot the median Gantt charts.

    :param end_results: the path to the end results
    :param algo: the algorithm
    :param dest: the directory
    :param source: the source directory
    :param best: should we plot the best instance only (or, otherwise, the
        median)
    :param insts: the instances to use
    """
    n: Final[str] = algorithm_namer(algo)
    plot_stat_gantt_charts(
        get_end_results(end_results, algos={algo},
                        insts=None if insts is None else set(insts)),
        name_base=sanitize_name(f"best_gantt_{n}" if best else f"gantt_{n}"),
        dest_dir=dest,
        results_dir=source,
        instance_sort_key=instance_sort_key,
        statistic=cast(
            Callable[[Iterable[int | float]], int | float],
            min if best else median))


def progress(algos: list[str], dest: Path, source: Path,
             log: bool = True, millis: bool = True) -> None:
    """
    Plot the median Gantt charts.

    :param algos: the algorithms
    :param dest: the directory
    :param source: the source directory
    :param log: is the time logarithmically scaled?
    :param millis: is the time measured in milliseconds?
    """
    n: str = f"progress_{algorithm_namer(algos[0])}_"
    if log:
        n = n + "log_"
    if millis:
        unit = TIME_UNIT_MILLIS
        n = n + "T"
    else:
        unit = TIME_UNIT_FES
        n = n + "FEs"
    plot_progresses(results_dir=source,
                    algorithms=algos,
                    name_base=n,
                    dest_dir=dest,
                    log_time=log,
                    time_unit=unit,
                    algorithm_namer=algorithm_namer,
                    instance_sort_key=instance_sort_key,
                    algorithm_sort_key=algorithm_sort_key)


def makespans_over_param(
        end_results: Path,
        selector: Callable[[str], bool],
        x_getter: Callable[[EndStatistics], int | float],
        name_base: str,
        dest_dir: str,
        x_axis: AxisRanger | Callable[[], AxisRanger]
        = AxisRanger,
        x_label: str | None = None,
        algo_getter: Callable[[str], str] | None = None,
        title: str | None = None,
        legend_pos: str = "right",
        title_x: float = 0.5,
        y_label_location: float = 1.0) -> list[Path]:
    """
    Plot the performance over a parameter.

    :param end_results: the end results path
    :param selector: the selector for algorithms
    :param name_base: the basic name
    :param dest_dir: the destination directory
    :param x_getter: the function computing the x-value for each statistics
        object
    :param x_axis: the axis ranger
    :param x_label: the x-axis label
    :param algo_getter: the optional algorithm name getter (use `name_base` if
        unspecified)
    :param title: the optional title (use `name_base` if unspecified)
    :param legend_pos: the legend position, set to "right"
    :param title_x: the title x
    :param y_label_location: the y label location
    :returns: the list of generated files
    """

    def _algo_name_getter(es: EndStatistics,
                          n=name_base, g=algo_getter) -> str:
        return n if g is None else g(es.algorithm)

    return plot_end_makespans_over_param(
        end_results=get_end_results(end_results, algos=selector),
        x_getter=x_getter, name_base=sanitize_name(f"{name_base}_results"),
        dest_dir=dest_dir, title=name_base if title is None else title,
        algorithm_getter=_algo_name_getter,  # type: ignore
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key,
        x_axis=x_axis, x_label=x_label,
        plot_single_instances=algo_getter is None,
        legend_pos=legend_pos, title_x=title_x,
        y_label_location=y_label_location)


def evaluate_experiment(results_dir: str = pp.join(".", "results"),
                        dest_dir: str | None = None) -> None:
    """
    Evaluate the experiment.

    :param results_dir: the results directory
    :param dest_dir: the destination directory
    """
    source: Final[Path] = directory_path(results_dir)
    dest: Final[Path] = Path(dest_dir if dest_dir else
                             pp.join(source, "..", "evaluation"))
    dest.ensure_dir_exists()
    logger(f"Beginning evaluation from {source!r} to {dest!r}.")

    end_results: Final[Path] = compute_end_results(source, dest)
    if not end_results:
        raise ValueError("End results path is empty??")
    end_stats: Final[Path] = compute_end_statistics(end_results, dest)
    if not end_stats:
        raise ValueError("End stats path is empty??")

    logger("Now evaluating the single random sampling algorithm `1rs`.")
    table(end_results, ["1rs"], dest)
    makespans(end_results, ["1rs"], dest)
    gantt(end_results, "1rs", dest, source)

    logger("Now evaluating the multi-random sampling algorithm `rs`.")
    table(end_results, ["rs", "1rs"], dest)
    makespans(end_results, ["rs", "1rs"], dest)
    gantt(end_results, "rs", dest, source)
    progress(["rs"], dest, source)
    progress(["rs"], dest, source, log=False)

    logger("Now evaluating the hill climbing algorithm `hc`.")
    table(end_results, ["hc_swap2", "rs"], dest)
    makespans(end_results, ["hc_swap2", "rs"], dest)
    gantt(end_results, "hc_swap2", dest, source)
    progress(["hc_swap2", "rs"], dest, source)
    progress(["hc_swap2", "rs"], dest, source, millis=False)

    logger(f"Finished evaluation from {source!r} to {dest!r}.")


# Evaluate experiment if run as script
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipy_argparser(
        __file__, "Evaluate the results of the JSSP example experiment",
        "This experiment evaluates all the results of the JSSP example"
        " experiment and creates the figures and tables of the "
        "'Optimization Algorithms' book (see "
        "http://thomasweise.github.io/oa).")
    parser.add_argument(
        "source", nargs="?", default="./results", type=Path,
        help="the directory with the results of the JSSP experiment")
    parser.add_argument(
        "dest", type=Path, nargs="?", default="./evaluation/",
        help="the directory to write the evaluation results to")
    args: Final[argparse.Namespace] = parser.parse_args()
    evaluate_experiment(results_dir=args.source, dest_dir=args.dest)
