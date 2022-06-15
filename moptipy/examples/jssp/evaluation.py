"""Evaluate the results of the example experiment."""
import os.path as pp
import sys
from statistics import median
from typing import Dict, Final, Optional, Any, List, Set, Union, Callable, \
    Iterable, cast

from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import TIME_UNIT_FES, TIME_UNIT_MILLIS
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_statistics import EndStatistics
from moptipy.evaluation.tabulate_end_results_impl import \
    tabulate_end_results, command_column_namer
from moptipy.examples.jssp.experiment import EXPERIMENT_INSTANCES
from moptipy.examples.jssp.plots import plot_end_makespans, \
    plot_stat_gantt_charts, plot_progresses, plot_end_makespans_over_param
from moptipy.utils.console import logger
from moptipy.utils.help import help_screen
from moptipy.utils.lang import EN
from moptipy.utils.path import Path
from moptipy.utils.types import type_error

#: The letter mu
LETTER_M: Final[str] = "\u03BC"
#: The letter lambda
LETTER_L: Final[str] = "\u03BB"

#: the pre-defined instance sort keys
__INST_SORT_KEYS: Final[Dict[str, int]] = {
    n: i for i, n in enumerate(EXPERIMENT_INSTANCES)
}


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


#: the pre-defined algorithm sort keys
__ALGO_SORT_KEYS: Final[Dict[str, int]] = {
    n: i for i, n in enumerate([
        "1rs", "rs", "hc", "hc_swap2", "hcr_32768_swap2", "hcr", "hcn",
        "hc_swapn", "hcr_65536_swapn", "hcrn", "rls", "rls_swap2",
        "rlsn", "rls_swapn", "rw", "rw_swap2", "rw_swapn"])
}


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


#: the algorithm name map
__ALGO_NAME_MAP: Final[Dict[str, str]] = {
    "hc_swap2": "hc", "hcr_32768_swap2": "hcr", "hc_swapn": "hcn",
    "hcr_65536_swapn": "hcrn", "rls_swap2": "rls", "rls_swapn": "rlsn"
}


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


def table(end_results: Path, algos: List[str], dest: Path) -> None:
    """
    Tabulate the end results.

    :param end_results: the path to the end results
    :param algos: the algorithms
    :param dest: the directory
    """
    EN.set_current()
    n: Final[str] = algorithm_namer(algos[0])
    tabulate_end_results(
        end_results=get_end_results(end_results, algos=set(algos)),
        file_name=f"end_results_{n}", dir_name=dest,
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key,
        col_namer=command_column_namer,
        algorithm_namer=algorithm_namer,
        use_lang=False)


def makespans(end_results: Path, algos: List[str], dest: Path) -> None:
    """
    Plot the end makespans.

    :param end_results: the path to the end results
    :param algos: the algorithms
    :param dest: the directory
    """
    n: Final[str] = algorithm_namer(algos[0])
    plot_end_makespans(
        end_results=get_end_results(end_results, algos=set(algos)),
        name_base=f"makespan_scaled_{n}", dest_dir=dest,
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key,
        algorithm_namer=algorithm_namer)


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
        name_base=f"best_gantt_{n}" if best else f"gantt_{n}",
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
        title: Optional[str] = None) -> List[Path]:
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
    :returns: the list of generated files
    """
    def _algo_name_getter(es: EndStatistics,
                          n=name_base, g=algo_getter) -> str:
        return n if g is None else g(es.algorithm)

    return plot_end_makespans_over_param(
        end_results=get_end_results(end_results, algos=selector),
        x_getter=x_getter, name_base=f"{name_base}_results",
        dest_dir=dest_dir, title=name_base if title is None else title,
        algorithm_getter=_algo_name_getter,  # type: ignore
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key,
        x_axis=x_axis, x_label=x_label,
        plot_single_instances=algo_getter is None)


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
