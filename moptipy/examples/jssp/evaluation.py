"""Evaluate the results of the example experiment."""
import os.path as pp
import sys
from typing import Dict, Final, Optional, Any, List, Tuple, Set, Iterable

from moptipy.evaluation.end_results import EndResult
from moptipy.examples.jssp.experiment import EXPERIMENT_INSTANCES
from moptipy.examples.jssp.plots import plot_end_makespans
from moptipy.utils.log import logger
from moptipy.utils.path import Path

#: the pre-defined instance sort keys
__INST_SORT_KEYS: Final[Dict[str, Tuple[int, str]]] = {
    n: (i, n) for i, n in enumerate(EXPERIMENT_INSTANCES)
}


def instance_sort_key(name: str) -> Tuple[int, str]:
    """
    Get the instance sort key for a given instance name.

    :param str name: the instance name
    :returns: the sort key
    :rtype: Tuple[int, str]
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be str, but is {type(name)}.")
    if not name:
        raise ValueError("name must not be empty.")
    if name not in __INST_SORT_KEYS:
        __INST_SORT_KEYS[name] = (1000, name)
    return __INST_SORT_KEYS[name]


def algorithm_sort_key(name: str) -> str:
    """
    Get the algorithm sort key for a given algorithm name.

    :param str name: the algorithm name
    :returns: the sort key
    :rtype: Tuple[int, str]
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be str, but is {type(name)}.")
    if not name:
        raise ValueError("name must not be empty.")
    return name


def compute_end_results(results_dir: str,
                        dest_dir: str) -> Path:
    """
    Get the end results, compute them if necessary.

    :param str results_dir: the results directory
    :param str dest_dir: the destination directory
    :returns: the path to the end results file.
    :rtype: Path
    """
    dest: Final[Path] = Path.directory(dest_dir)
    results_file: Final[Path] = dest.resolve_inside("end_results.txt")
    if results_file.is_file():
        return results_file
    results: Final[List[EndResult]] = []

    source: Final[Path] = Path.directory(results_dir)
    logger(f"loading end results from path '{source}'.")
    EndResult.from_logs(source, results)
    if not results:
        raise ValueError(f"Could not find any logs in '{source}'.")
    results.sort()
    logger(f"found {len(results)} log files in path '{source}', storing "
           f"them in file '{results_file}'.")
    EndResult.to_csv(results, results_file)
    results_file.enforce_file()
    logger(f"finished writing file '{results_file}'.")
    return results_file


def get_end_results(file: str,
                    insts: Optional[Set[str]] = None,
                    algos: Optional[Set[str]] = None) -> Iterable[EndResult]:
    """
    Get a specific set of end results..

    :param str file: the end results file
    :param Optional[Set[str]] insts: only these instances will be included if
        this parameter is provided
    :param Optional[Set[str]] algos: only these algorithms will be included if
        this parameter is provided
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
    EndResult.from_csv(file=file, collector=col, filterer=__filter)
    if len(col) <= 0:
        raise ValueError(
            f"no end results for instances {insts} and algorithms {algos}.")
    return col


def evaluate_experiment(results_dir: str = pp.join(".", "results"),
                        dest_dir: Optional[str] = None) -> None:
    """
    Evaluate the experiment.

    :param str results_dir: the results directory
    :param str dest_dir: the destination directory
    """
    source: Final[Path] = Path.directory(results_dir)
    dest: Final[Path] = Path.path(dest_dir if dest_dir else
                                  pp.join(source, "..", "evaluation"))
    dest.ensure_dir_exists()
    logger(f"Beginning evaluation from '{source}' to '{dest}'.")

    end_results: Final[Path] = compute_end_results(source, dest)
    if not end_results:
        raise ValueError("End results path is empty??")

    plot_end_makespans(
        end_results=get_end_results(end_results, algos={"1rs"}),
        name_base="makespan_standardized_1rs",
        dest_dir=dest,
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key)

    plot_end_makespans(
        end_results=get_end_results(end_results, algos={"1rs", "rs"}),
        name_base="makespan_standardized_rs",
        dest_dir=dest,
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key)

    plot_end_makespans(
        end_results=get_end_results(end_results, algos={"rs", "hc_swap2"}),
        name_base="makespan_standardized_hc_swap2",
        dest_dir=dest,
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key)

    plot_end_makespans(
        end_results=get_end_results(end_results,
                                    algos={"rs", "hc_swap2", "ea1p1_swap2"}),
        name_base="makespan_standardized_ea1p1_swap2",
        dest_dir=dest,
        instance_sort_key=instance_sort_key,
        algorithm_sort_key=algorithm_sort_key)

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
