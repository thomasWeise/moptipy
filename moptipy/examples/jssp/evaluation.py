"""Evaluate the results of the example experiment."""
import os.path as pp
import sys
from re import compile as _compile, match, sub
from statistics import median
from typing import Dict, Final, Optional, Any, List, Set, Union, Callable, \
    Iterable, cast, Tuple

from math import log2
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import TIME_UNIT_FES, TIME_UNIT_MILLIS
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_statistics import EndStatistics
from moptipy.evaluation.tabulate_end_results import \
    tabulate_end_results, command_column_namer, \
    DEFAULT_ALGORITHM_INSTANCE_STATISTICS, \
    DEFAULT_ALGORITHM_SUMMARY_STATISTICS
from moptipy.examples.jssp.experiment import DEFAULT_ALGORITHMS
from moptipy.examples.jssp.experiment import EXPERIMENT_INSTANCES
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.plots import plot_end_makespans, \
    plot_stat_gantt_charts, plot_progresses, plot_end_makespans_over_param
from moptipy.spaces.permutations import Permutations
from moptipy.utils.console import logger
from moptipy.api.logging import KEY_TOTAL_FES,\
    KEY_LAST_IMPROVEMENT_TIME_MILLIS
from moptipy.evaluation.statistics import KEY_MEAN_ARITH
from moptipy.utils.help import help_screen
from moptipy.utils.lang import EN
from moptipy.utils.path import Path
from moptipy.utils.types import type_error
from moptipy.utils.logger import sanitize_name, SCOPE_SEPARATOR

#: The letter mu
LETTER_M: Final[str] = "\u03BC"
#: The letter lambda
LETTER_L: Final[str] = "\u03BB"

#: the pre-defined instance sort keys
__INST_SORT_KEYS: Final[Dict[str, int]] = {
    __n: __i for __i, __n in enumerate(EXPERIMENT_INSTANCES)
}


#: the name of the mu+1 EA
NAME_EA_MU_PLUS_1: Final[str] = f"{LETTER_M}+1_ea"
#: the name of the mu+mu EA
NAME_EA_MU_PLUS_MU: Final[str] = f"{LETTER_M}+{LETTER_M}_ea"
#: the name of the mu+ld(mu) EA
NAME_EA_MU_PLUS_LOG_MU: Final[str] = f"{LETTER_M}+log\u2082{LETTER_M}_ea"
#: the name of the mu+sqrt(mu) EA
NAME_EA_MU_PLUS_SQRT_MU: Final[str] = f"{LETTER_M}+\u221A{LETTER_M}_ea"


def ea_family(name: str) -> str:
    """
    Name the evolutionary algorithm setup.

    :param name: the full name
    :returns: the short name of the family
    """
    if name.startswith("ea_"):
        ss = name.split("_")
        if len(ss) == 4:
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
    raise ValueError(f"Invalid name '{name}'.")


def __make_algo_names() -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Create the algorithm sort keys and name map.

    :returns: the algorithm sort keys and name map
    """
    inst = Instance.from_resource("demo")
    space = Permutations.with_repetitions(inst.jobs, inst.machines)
    names: List[str] = [str(a(inst, space)) for a in DEFAULT_ALGORITHMS]
    names_new = list(names)

    namer: Dict[str, str] = {}
    used_names: Set[str] = set(names)
    for pattern, repl in [("hc_swap2", "hc"),
                          ("hcr_32768_swap2", "hcr"),
                          ("hc_swapn", "hcn"),
                          ("hcr_65536_swapn", "hcrn"),
                          ("rls_swap2", "rls"),
                          ("rls_swapn", "rlsn"),
                          ("ea_([0-9]+)_([0-9]+)_swap2", "\\1+\\2_ea")]:
        re = _compile(pattern)
        found = False
        for s in names:
            m = match(re, s)
            if m is not None:
                ns = sub(re, repl, s)
                if (ns is None) or (len(ns) <= 0):
                    raise ValueError(f"'{s}' -> '{ns}'?")
                if ns == s:
                    continue
                found = True
                os = namer.get(s, None)
                if os is not None:
                    if os == ns:
                        continue
                    raise ValueError(f"'{s}' -> '{ns}', '{os}'?")
                if ns in used_names:
                    raise ValueError(f"Already got '{ns}'.")
                namer[s] = ns
                names_new.insert(names_new.index(s), ns)
        if not found:
            raise ValueError(f"did not find '{pattern}'.")

    ea1p1: Final[int] = names_new.index("ea_1_1_swap2")
    ea_families: Dict[str, int] = {
        NAME_EA_MU_PLUS_MU: ea1p1,
        NAME_EA_MU_PLUS_SQRT_MU: ea1p1,
        NAME_EA_MU_PLUS_LOG_MU: ea1p1,
        NAME_EA_MU_PLUS_1: ea1p1}
    for n in names:
        if n.startswith("ea_"):
            try:
                fam = ea_family(n)
            except ValueError:
                continue
            if fam not in ea_families:
                if fam in used_names:
                    raise ValueError(f"duplicated ea family '{fam}'.")
                used_names.add(fam)
                ea_families[fam] = names_new.index(n)
            else:
                ea_families[fam] = min(ea_families[fam], names_new.index(n))
    for i, n in sorted([(n[1], n[0]) for n in ea_families.items()],
                       reverse=True, key=lambda a: a[0]):
        names_new.insert(i, n)

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
    dest: Final[Path] = Path.directory(dest_dir)
    results_file: Final[Path] = dest.resolve_inside("end_results.txt")
    if results_file.is_file():
        return results_file
    results: Final[List[EndResult]] = []

    source: Final[Path] = Path.directory(results_dir)
    logger(f"loading end results from path '{source}'.")
    EndResult.from_logs(source, results.append)
    if not results:
        raise ValueError(f"Could not find any logs in '{source}'.")
    results.sort()
    logger(f"found {len(results)} log files in path '{source}', storing "
           f"them in file '{results_file}'.")
    rf: Path = EndResult.to_csv(results, results_file)
    if rf != results_file:
        raise ValueError(
            f"results file should be {results_file}, but is {rf}.")
    results_file.enforce_file()
    logger(f"finished writing file '{results_file}'.")
    return results_file


def get_end_results(
        file: str,
        insts: Union[None, Set[str], Callable[[str], bool]] = None,
        algos: Union[None, Set[str], Callable[[str], bool]] = None) \
        -> List[EndResult]:
    """
    Get a specific set of end results..

    :param file: the end results file
    :param insts: only these instances will be included if this parameter is
        provided
    :param algos: only these algorithms will be included if this parameter is
        provided
    """
    def __filter(er: EndResult,
                 ins=None if insts is None else
                 insts.__contains__ if isinstance(insts, Set)
                 else insts,
                 alg=None if algos is None else
                 algos.__contains__ if isinstance(algos, Set)
                 else algos) -> bool:
        if ins is not None:
            if not ins(er.instance):
                return False
        if alg is not None:
            if not alg(er.algorithm):
                return False
        return True

    col: Final[List[EndResult]] = []
    EndResult.from_csv(file=file, consumer=col.append, filterer=__filter)
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
    dest: Final[Path] = Path.directory(dest_dir)
    stats_file: Final[Path] = dest.resolve_inside("end_statistics.txt")
    if stats_file.is_file():
        return stats_file

    results: Final[List[EndResult]] = get_end_results(end_results_file)
    if len(results) <= 0:
        raise ValueError("end results cannot be empty")
    stats: Final[List[EndStatistics]] = []
    EndStatistics.from_end_results(results, stats.append)
    if len(stats) <= 0:
        raise ValueError("end result statistics cannot be empty")
    stats.sort()

    sf: Path = EndStatistics.to_csv(stats, stats_file)
    if sf != stats_file:
        raise ValueError(f"stats file should be {stats_file} but is {sf}")
    stats_file.enforce_file()
    logger(f"finished writing file '{stats_file}'.")
    return stats_file


def table(end_results: Path, algos: List[str], dest: Path,
          swap_stats: Optional[Iterable[Tuple[str, str]]] = None) -> None:
    """
    Tabulate the end results.

    :param end_results: the path to the end results
    :param algos: the algorithms
    :param dest: the directory
    :param swap_stats: the statistics to swap out
    """
    EN.set_current()
    n: Final[str] = algorithm_namer(algos[0])
    algo_inst_stat: Union[List, Tuple] = DEFAULT_ALGORITHM_INSTANCE_STATISTICS
    algo_sum_stat: Union[List, Tuple] = DEFAULT_ALGORITHM_SUMMARY_STATISTICS
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
                raise ValueError(f"did not find '{old}' in {lst}.")

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


def makespans(end_results: Path, algos: List[str], dest: Path,
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
          insts: Optional[Iterable[str]] = None) -> None:
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
            Callable[[Iterable[Union[int, float]]], Union[int, float]],
            min if best else median))


def progress(algos: List[str], dest: Path, source: Path,
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
        x_getter: Callable[[EndStatistics], Union[int, float]],
        name_base: str,
        dest_dir: str,
        x_axis: Union[AxisRanger, Callable[[], AxisRanger]]
        = AxisRanger,
        x_label: Optional[str] = None,
        algo_getter: Optional[Callable[[str], str]] = None,
        title: Optional[str] = None,
        legend_pos: str = "right",
        title_x: float = 0.5) -> List[Path]:
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
        legend_pos=legend_pos, title_x=title_x)


def evaluate_experiment(results_dir: str = pp.join(".", "results"),
                        dest_dir: Optional[str] = None) -> None:
    """
    Evaluate the experiment.

    :param results_dir: the results directory
    :param dest_dir: the destination directory
    """
    source: Final[Path] = Path.directory(results_dir)
    dest: Final[Path] = Path.path(dest_dir if dest_dir else
                                  pp.join(source, "..", "evaluation"))
    dest.ensure_dir_exists()
    logger(f"Beginning evaluation from '{source}' to '{dest}'.")

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

    logger("Now evaluating the hill climbing algorithm with "
           "restarts 'hcr' on 'swap2'.")
    makespans_over_param(
        end_results,
        lambda an: an.startswith("hcr_") and an.endswith("_swap2"),
        lambda es: int(es.algorithm.split("_")[1]),
        "hcr_L_swap2", dest,
        lambda: AxisRanger(log_scale=True, log_base=2.0), "L")
    table(end_results, ["hcr_32768_swap2", "hc_swap2", "rs"], dest)
    makespans(end_results, ["hcr_32768_swap2", "hc_swap2", "rs"], dest)
    gantt(end_results, "hcr_32768_swap2", dest, source)
    progress(["hcr_32768_swap2", "hc_swap2", "rs"], dest, source)

    logger("Now evaluating the hill climbing algorithm with 'swapn'.")
    table(end_results, ["hc_swapn", "hcr_32768_swap2", "hc_swap2"], dest)
    makespans(end_results, ["hc_swapn", "hcr_32768_swap2", "hc_swap2"], dest)
    progress(["hc_swapn", "hcr_32768_swap2", "hc_swap2"], dest, source)
    progress(["hc_swapn", "hcr_32768_swap2", "hc_swap2"], dest, source,
             millis=False)

    logger("Now evaluating the hill climbing algorithm with "
           "restarts 'hcr' on 'swapn'.")
    makespans_over_param(
        end_results,
        lambda an: an.startswith("hcr_") and an.endswith("_swapn"),
        lambda es: int(es.algorithm.split("_")[1]),
        "hcr_L_swapn", dest,
        lambda: AxisRanger(log_scale=True, log_base=2.0), "L")
    table(end_results, ["hcr_65536_swapn", "hc_swapn",
                        "hcr_32768_swap2"], dest)
    makespans(end_results, ["hcr_65536_swapn", "hc_swapn",
                            "hcr_32768_swap2"], dest)
    progress(["hcr_65536_swapn", "hc_swapn",
              "hcr_32768_swap2"], dest, source)

    logger("Now evaluating the RLS algorithm with 'swap2' and 'swapn'.")
    table(end_results, ["rls_swapn", "rls_swap2",
                        "hcr_32768_swap2", "hcr_65536_swapn"], dest)
    makespans(end_results, ["rls_swapn", "rls_swap2", "hcr_32768_swap2",
                            "hcr_65536_swapn"], dest)
    gantt(end_results, "rls_swap2", dest, source)
    progress(["rls_swapn", "rls_swap2", "hcr_32768_swap2",
              "hcr_65536_swapn"], dest, source)
    gantt(end_results, "rls_swap2", dest, source, True, ["ta70"])

    logger("Now evaluating EA without crossover.")

    makespans_over_param(
        end_results=end_results,
        selector=lambda n: n.startswith("ea_") and n.split("_")[3] == "swap2",
        x_getter=lambda es: int(es.algorithm.split("_")[1]),
        name_base="ea_no_cr",
        algo_getter=ea_family,
        title=f"{LETTER_M}+{LETTER_L}_ea", x_label=LETTER_M,
        title_x=0.8,
        x_axis=AxisRanger(log_scale=True, log_base=2.0),
        legend_pos="upper center",
        dest_dir=dest)
    lims: Final[str] = f"{KEY_LAST_IMPROVEMENT_TIME_MILLIS}{SCOPE_SEPARATOR}" \
                       f"{KEY_MEAN_ARITH}"
    totfes: Final[str] = f"{KEY_TOTAL_FES}{SCOPE_SEPARATOR}{KEY_MEAN_ARITH}"
    table(end_results, ["ea_1_2_swap2", "ea_2_2_swap2", "ea_2_4_swap2",
                        "ea_512_512_swap2", "rls_swap2"], dest,
          swap_stats=[(lims, totfes)])
    makespans(end_results, ["ea_1_2_swap2", "ea_2_2_swap2",
                            "ea_512_512_swap2", "rls_swap2"], dest, 0.6)
    progress(["ea_1_2_swap2", "ea_2_2_swap2", "ea_2_4_swap2",
              "ea_64_1_swap2", "ea_1024_1024_swap2", "ea_8192_65536_swap2",
              "rls_swap2"],
             dest, source)

    logger(f"Finished evaluation from '{source}' to '{dest}'.")


# Evaluate experiment if run as script
if __name__ == '__main__':
    help_screen(
        "JSSP Experiment Evaluator", __file__,
        "Evaluate the results of the JSSP experiment.",
        [("results_dir",
          "the directory with the results of the JSSP experiment",
          True),
         ("dest_dir",
          "the directory where the evaluation results should be stored",
          True)])
    mkwargs: Dict[str, Any] = {}
    if len(sys.argv) > 1:
        results_dir_str: Final[str] = sys.argv[1]
        mkwargs["results_dir"] = results_dir_str
        logger(f"Set results_dir '{results_dir_str}'.")
        if len(sys.argv) > 2:
            dest_dir_str: Final[str] = sys.argv[2]
            mkwargs["dest_dir"] = dest_dir_str
            logger(f"Set dest_dir to '{dest_dir_str}'.")

    evaluate_experiment(**mkwargs)  # invoke the experiment evaluation
