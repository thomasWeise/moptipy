"""Evaluate the results of the example experiment."""
import os.path as pp
import sys
from typing import Dict, Final, Optional, Any, List, Set

from moptipy.api.logging import KEY_LAST_IMPROVEMENT_TIME_MILLIS, \
    KEY_TOTAL_TIME_MILLIS
from moptipy.evaluation.end_results import EndResult, KEY_FES_PER_MS
from moptipy.evaluation.end_statistics import EndStatistics
from moptipy.evaluation.tabulate_end_results_impl import \
    DEFAULT_ALGORITHM_INSTANCE_STATISTICS, \
    DEFAULT_ALGORITHM_SUMMARY_STATISTICS, \
    tabulate_end_results, command_column_namer
from moptipy.examples.jssp.experiment import EXPERIMENT_INSTANCES
from moptipy.examples.jssp.plots import plot_end_makespans, \
    plot_median_gantt_charts, plot_progresses
from moptipy.utils.console import logger
from moptipy.utils.lang import Lang
from moptipy.utils.path import Path
from moptipy.utils.types import type_error


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
    n: i for i, n in enumerate(["1rs", "rs", "hc", "rls", "rw"])
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
    "hc_swap2": "hc", "rls_swap2": "rls", "rw_swap2": "rw"
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


def get_end_results(file: str,
                    insts: Optional[Set[str]] = None,
                    algos: Optional[Set[str]] = None) -> List[EndResult]:
    """
    Get a specific set of end results..

    :param file: the end results file
    :param insts: only these instances will be included if this parameter is
        provided
    :param algos: only these algorithms will be included if this parameter is
        provided
    """
    def __filter(er: EndResult, ins=insts, alg=algos) -> bool:
        if ins is not None:
            if er.instance not in ins:
                return False
        if alg is not None:
            if er.algorithm not in alg:
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
    for lang in Lang.all():
        lang.set_current()
        tabulate_end_results(
            end_results=get_end_results(end_results, algos={"1rs"}),
            file_name="end_results_1rs",
            dir_name=dest,
            algorithm_instance_statistics=[
                c.replace(KEY_LAST_IMPROVEMENT_TIME_MILLIS,
                          KEY_TOTAL_TIME_MILLIS)
                for c in DEFAULT_ALGORITHM_INSTANCE_STATISTICS
                if KEY_FES_PER_MS not in c],
            algorithm_summary_statistics=[
                c.replace(KEY_LAST_IMPROVEMENT_TIME_MILLIS,
                          KEY_TOTAL_TIME_MILLIS)
                for c in DEFAULT_ALGORITHM_SUMMARY_STATISTICS
                if KEY_FES_PER_MS not in c],
            instance_sort_key=instance_sort_key,
            algorithm_sort_key=algorithm_sort_key,
            col_namer=command_column_namer)
    plot_end_makespans(
        end_results=get_end_results(end_results, algos={"1rs"}),
        name_base="makespan_scaled_1rs",
        dest_dir=dest,
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key)
    plot_median_gantt_charts(get_end_results(end_results, algos={"1rs"}),
                             name_base="gantt_1rs",
                             dest_dir=dest,
                             results_dir=source,
                             instance_sort_key=instance_sort_key)

    logger("Now evaluating the multi-random sampling algorithm `rs`.")
    for lang in Lang.all():
        lang.set_current()
        tabulate_end_results(
            end_results=get_end_results(end_results, algos={"1rs", "rs"}),
            file_name="end_results_rs",
            dir_name=dest,
            instance_sort_key=instance_sort_key,
            algorithm_sort_key=algorithm_sort_key,
            col_namer=command_column_namer)
    plot_end_makespans(
        end_results=get_end_results(end_results, algos={"1rs", "rs"}),
        name_base="makespan_scaled_rs",
        dest_dir=dest,
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key)
    plot_median_gantt_charts(get_end_results(end_results, algos={"rs"}),
                             name_base="gantt_rs",
                             dest_dir=dest,
                             results_dir=source,
                             instance_sort_key=instance_sort_key)
    plot_progresses(results_dir=source,
                    algorithms=["rs"],
                    name_base="progress_rs",
                    dest_dir=dest,
                    log_time=False)
    plot_progresses(results_dir=source,
                    algorithms=["rs"],
                    name_base="progress_rs_log_T",
                    dest_dir=dest,
                    log_time=True)

    logger("Now evaluating the hill climbing algorithm `hc`.")
    for lang in Lang.all():
        lang.set_current()
        tabulate_end_results(
            end_results=get_end_results(end_results, algos={"rs", "hc_swap2"}),
            file_name="end_results_hc",
            dir_name=dest,
            instance_sort_key=instance_sort_key,
            algorithm_sort_key=algorithm_sort_key,
            col_namer=command_column_namer)
    plot_end_makespans(
        end_results=get_end_results(end_results, algos={"rs", "hc_swap2"}),
        name_base="makespan_scaled_hc",
        dest_dir=dest,
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key,
        algorithm_namer=algorithm_namer)
    plot_median_gantt_charts(get_end_results(end_results, algos={"hc"}),
                             name_base="gantt_hc",
                             dest_dir=dest,
                             results_dir=source,
                             instance_sort_key=instance_sort_key)
    plot_progresses(results_dir=source,
                    algorithms=["rs", "hc"],
                    name_base="progress_rs_log_T",
                    dest_dir=dest,
                    log_time=True,
                    algorithm_namer=algorithm_namer)

    logger(f"Finished evaluation from '{source}' to '{dest}'.")


# Evaluate experiment if run as script
if __name__ == '__main__':
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
