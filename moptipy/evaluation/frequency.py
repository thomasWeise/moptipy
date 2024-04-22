"""
Load the encounter frequencies or the set of different objective values.

This tool can load the different objective values that exist or are
encountered by optimization processes. This may be useful for statistical
evaluations or fitness landscape analyses.

This tool is based on code developed by Mr. Tianyu LIANG (梁天宇),
MSc student at the Institute of Applied Optimization (IAO,
应用优化研究所) of the School of Artificial Intelligence and
Big Data (人工智能与大数据学院) of Hefei University (合肥学院).
"""
import argparse
import os.path
from collections import Counter
from gc import collect
from math import isfinite
from typing import Any, Callable, Final, Iterable

from pycommons.io.console import logger
from pycommons.io.csv import CSV_SEPARATOR, SCOPE_SEPARATOR
from pycommons.io.path import Path
from pycommons.strings.string_conv import str_to_num
from pycommons.types import type_error

from moptipy.algorithms.so.fea1plus1 import H_LOG_SECTION
from moptipy.api.logging import (
    KEY_F_LOWER_BOUND,
    KEY_F_UPPER_BOUND,
    KEY_INSTANCE,
    PROGRESS_CURRENT_F,
    SCOPE_OBJECTIVE_FUNCTION,
    SECTION_PROGRESS,
)
from moptipy.evaluation.base import (
    MultiRunData,
    PerRunData,
)
from moptipy.evaluation.log_parser import SetupAndStateParser
from moptipy.utils.help import moptipy_argparser

#: the lower bound of the objective function
_FULL_KEY_LOWER_BOUND: Final[str] = \
    f"{SCOPE_OBJECTIVE_FUNCTION}{SCOPE_SEPARATOR}{KEY_F_LOWER_BOUND}"
#: the upper bound of the objective function
_FULL_KEY_UPPER_BOUND: Final[str] = \
    f"{SCOPE_OBJECTIVE_FUNCTION}{SCOPE_SEPARATOR}{KEY_F_UPPER_BOUND}"


def from_logs(
        path: str, consumer: Callable[[PerRunData, Counter[
            int | float]], Any],
        report_progress: bool = True,
        report_lower_bound: bool = False,
        report_upper_bound: bool = False,
        report_goal_f: bool = False,
        report_h: bool = True,
        per_instance_known: Callable[[
            str], Iterable[int | float]] = lambda _: ()) -> None:
    """
    Parse a path, pass all discovered objective values per-run to a consumer.

    This function parses the log files in a directory recursively. For each
    log file, it produces a `Counter` filled with all encountered objective
    values and their "pseudo" encounter frequencies. "pseudo" because the
    values returned depend very much on how the function is configured.

    First, if all other parameters are set to `False`, the function passes
    a `Counter` to the `consumer` where the best encountered objective value
    has frequency `1` and no other data is present.
    If `report_progress` is `True`, then each time any objective value is
    encountered in the `PROGRESS` section, its counter is incremented by `1`.
    If the `PROGRESS` section is present, that it. The best encountered
    objective value will have a count of at least one either way.
    If `report_goal_f`, `report_lower_bound`, or `report_upper_bound` are
    `True`, then it is ensured that the goal objective value of the
    optimization process, the lower bound of the objective function, or
    the upper bound of the objective function will have a corresponding count
    of at least `1` if they are present in the log files (in the `SETUP`
    section).
    If `report_h` is `True`, then a frequency fitness assignment `H` section
    is parsed, if present (see :mod:`~moptipy.algorithms.so.fea1plus1`). Such
    a section contains tuples of objective values and encounter frequencies.
    These encounter frequencies are added to the counter. This means that if
    you set *both* `report_progress` and `report_h` to `True`, you will get
    frequencies that are too high.
    Finally, the function `per_instance_known` may return a set of known
    objective values for a given instance (based on its parameter, the
    instance name). Each such objective value will have a frequency of at
    least `1`.

    Generally, if we want the actual encounter frequencies of objective
    values, we could log *all FEs* to the log files and set `report_progress`
    to `True` and everything else to `False`. Then we get correct encounter
    frequencies. Alternatively, if we have a purly FFA-based algorithm (see,
    again, :mod:`~moptipy.algorithms.so.fea1plus1`), then we can set
    `report_progress` to `True` and everything else to `False` to get a
    similar result, but the encounter frequencies then depend on the selection
    scheme. Alternatively, if we only care about whether an objective value
    was encountered or not, we can simply set both to `True`. Finally, if we
    want to get all *possible* objective values, then we may also set
    `report_goal_f`, `report_lower_bound`, or `report_upper_bound` to `True`
    **if** we are sure that the corresponding objective values do actually
    exist (and are not just bounds that can never be reached).

    :param path: the path to parse
    :param consumer: the consumer receiving, for each log file, an instance of
        :class:`~moptipy.evaluation.base.PerRunData` identifying the run and
        a dictionary with the objective values and lower bounds of their
        existence or encounter frequency. Warning: The dictionary will be
        cleared and re-used for all files.
    :param report_progress: should all values in the `PROGRESS` section be
        reported, if such section exists?
    :param report_lower_bound: should the lower bound reported, if any lower
        bound for the objective function is listed?
    :param report_upper_bound: should the upper bound reported, if any upper
        bound for the objective function is listed?
    :param report_h: should all values in the `H` section be reported, if such
        section exists?
    :param report_goal_f: should we report the goal objective value, if it is
        specified?
    :param per_instance_known: a function that returns a set of known
        objective values per instance
    """
    __InnerLogParser(consumer, report_progress, report_lower_bound,
                     report_upper_bound, report_h, report_goal_f,
                     per_instance_known).parse(path)


def aggregate_from_logs(
        path: str, consumer: Callable[[MultiRunData, Counter[
            int | float]], Any],
        per_instance: bool = True,
        per_algorithm_instance: bool = True,
        report_progress: bool = True,
        report_lower_bound: bool = False,
        report_upper_bound: bool = False,
        report_goal_f: bool = False,
        report_h: bool = True,
        per_instance_known: Callable[[
            str], Iterable[int | float]] = lambda _: ()) -> None:
    """
    Parse a path, aggregate all discovered objective values to a consumer.

    A version of :func:`from_logs` that aggregates results on a per-instance
    and/or per-algorithm-instance combination. The basic process of loading
    the data is described in :func:`from_logs`.

    :param path: the path to parse
    :param consumer: the consumer receiving the aggregated results
    :param per_instance: pass results to the consumer that are aggregated over
        all algorithms and setups and runs for a given instance
    :param per_algorithm_instance: pass results to the consumer that are
        aggregated over all runs and setups for a given algorithm-instance
        combination
    :param report_progress: see :func:`from_logs`
    :param report_lower_bound: see :func:`from_logs`
    :param report_upper_bound: see :func:`from_logs`
    :param report_h: see :func:`from_logs`
    :param report_goal_f: see :func:`from_logs`
    :param per_instance_known: see :func:`from_logs`
    """
    if not callable(consumer):
        raise type_error(consumer, "consumer", call=True)
    if not isinstance(per_instance, bool):
        raise type_error(per_instance, "per_instance", bool)
    if not isinstance(per_algorithm_instance, bool):
        raise type_error(
            per_algorithm_instance, "per_algorithm_instance", bool)

    collection: Final[dict[tuple[str, int, str], Counter[int | float]]] = {}
    runs: Final[Counter[tuple[str, int, str]]] = Counter()

    def __consume(d: PerRunData, c: Counter[int | float]) -> None:
        nonlocal collection
        nonlocal runs
        nonlocal per_instance
        nonlocal per_algorithm_instance

        inst: Final[str] = d.instance
        ai: tuple[str, int, str]
        if per_instance:
            ai = (inst, 1, "")
            runs[ai] += 1
            if ai in collection:
                collection[ai] += c
            else:
                collection[ai] = Counter(c)
        if per_algorithm_instance:
            ai = (inst, 0, d.algorithm)
            runs[ai] += 1
            if ai in collection:
                collection[ai] += c
            else:
                collection[ai] = Counter(c)

    from_logs(path, __consume, report_progress, report_lower_bound,
              report_upper_bound, report_goal_f, report_h, per_instance_known)

    for key in sorted(collection.keys()):
        val = collection[key]
        del collection[key]
        consumer(MultiRunData(key[2] if key[2] else None,
                              key[0], None, None, runs[key]), val)
        del val


def number_of_objective_values_to_csv(
        input_dir: str, output_file: str,
        per_instance: bool = True,
        per_algorithm_instance: bool = True,
        report_lower_bound: bool = False,
        report_upper_bound: bool = False,
        report_goal_f: bool = False,
        per_instance_known: Callable[[
            str], Iterable[int | float]] = lambda _: ()) -> None:
    """
    Print the number of unique objective values to a CSV file.

    A version of :func:`aggregate_from_logs` that collects the existing
    objective values and prints an overview to a file.

    :param input_dir: the path to parse
    :param output_file: the output file to generate
    :param per_instance: pass results to the consumer that are aggregated over
        all algorithms and setups and runs for a given instance
    :param per_algorithm_instance: pass results to the consumer that are
        aggregated over all runs and setups for a given algorithm-instance
        combination
    :param report_lower_bound: see :func:`from_logs`
    :param report_upper_bound: see :func:`from_logs`
    :param report_goal_f: see :func:`from_logs`
    :param per_instance_known: see :func:`from_logs`
    """
    input_path: Final[Path] = Path(input_dir)
    output_path: Final[Path] = Path(output_file)
    logger(f"Collecting number of objective values from {input_path!r} "
           f"in {output_path!r}.")
    logger(f"Lower bounds will{'' if report_lower_bound else ' not'} be "
           "treated as existing objective values.")
    logger(f"Upper bounds wil {'' if report_upper_bound else ' not'} be "
           "treated as existing objective values.")
    logger("Goal objective values bounds will"
           f"{'' if report_upper_bound else ' not'} be treated as existing "
           "objective values.")
    Path(os.path.dirname(output_path)).ensure_dir_exists()
    data: Counter[tuple[str, str]] = Counter()
    instances: set[str] = set()
    algorithms: set[str] = set()

    def __collector(d: MultiRunData, c: Counter[int | float]) -> None:
        nonlocal data
        nonlocal instances
        nonlocal algorithms
        inst: str = ""
        if d.instance is not None:
            inst = d.instance
            instances.add(inst)
        algo: str = ""
        if d.algorithm is not None:
            algo = d.algorithm
            algorithms.add(algo)
        data[(inst, algo)] += len(c)

    aggregate_from_logs(
        input_path, __collector, per_instance, per_algorithm_instance,
        True, report_lower_bound, report_upper_bound, report_goal_f, True,
        per_instance_known)

    algos: Final[list[str]] = sorted(algorithms)

    logger(f"Now writing information gathered for {len(algos)} "
           f"algorithms and {len(instances)} to {output_path!r}.")
    with output_path.open_for_write() as ow:
        wrt = ow.write

        # write the header
        wrt(KEY_INSTANCE)
        for a in algos:
            wrt(CSV_SEPARATOR)
            wrt(a)
        wrt(CSV_SEPARATOR)
        wrt("all")
        wrt("\n")

        # write the output
        for instance in sorted(instances):
            wrt(instance)
            for a in algos:
                wrt(CSV_SEPARATOR)
                wrt(str(data[(instance, a)]))
            wrt(CSV_SEPARATOR)
            wrt(str(data[(instance, "")]))
            wrt("\n")
    logger(f"Finished writing {output_path!r}.")


class __InnerLogParser(SetupAndStateParser):
    """The internal log parser class for gathering objective values."""

    def __init__(self,
                 consumer: Callable[[PerRunData, Counter[int | float]], Any],
                 report_progress: bool, report_lower_bound: bool,
                 report_upper_bound: bool, report_h: bool,
                 report_goal_f: bool, per_instance_known: Callable[[
                     str], Iterable[int | float]]):
        """
        Create the internal log parser.

        :param consumer: the consumer
        :param report_progress: should all values in the `PROGRESS` section be
            reported, if such section exists?
        :param report_lower_bound: should the lower bound reported, if any
            lower bound for the objective function is listed?
        :param report_upper_bound: should the upper bound reported, if any
            upper bound for the objective function is listed?
        :param report_h: should all values in the `H` section be reported, if
            such section exists?
        :param report_goal_f: should we report the goal objective value, if it
            is specified?
        :param per_instance_known: a function that returns a set of known
            objective values per instance
        """
        super().__init__()
        if not callable(consumer):
            raise type_error(consumer, "consumer", call=True)
        if not isinstance(report_progress, bool):
            raise type_error(report_progress, "report_progress", bool)
        if not isinstance(report_lower_bound, bool):
            raise type_error(report_lower_bound, "report_lower_bound", bool)
        if not isinstance(report_upper_bound, bool):
            raise type_error(report_upper_bound, "report_upper_bound", bool)
        if not isinstance(report_h, bool):
            raise type_error(report_h, "report_h", bool)
        if not isinstance(report_goal_f, bool):
            raise type_error(report_goal_f, "report_goal_f", bool)
        if not callable(per_instance_known):
            raise type_error(
                per_instance_known, "per_instance_known", call=True)
        #: the consumer
        self.__consumer: Final[Callable[[PerRunData, Counter], Any]] = consumer
        #: report the progress
        self.__report_progress: Final[bool] = report_progress
        #: report the frequency history
        self.__report_h: Final[bool] = report_h
        #: report the lower bound, if any
        self.__report_lower_bound: Final[bool] = report_lower_bound
        #: report the upper bound, if any
        self.__report_upper_bound: Final[bool] = report_upper_bound
        #: report the goal objective value
        self.__report_goal_f: Final[bool] = report_goal_f
        #: the per-instance known objective values
        self.__per_instance_known: Final[Callable[
            [str], Iterable[int | float]]] = per_instance_known
        #: the internal counter
        self.__counter: Final[Counter[int | float]] = Counter()
        #: the internal state variable
        self.__state_progress: int = 0
        #: the internal state variable
        self.__state_h: int = 0

    def start_file(self, path: Path) -> bool:
        """
        Begin parsing the file identified by `path`.

        :param path: the path identifying the file
        """
        if not super().start_file(path):
            return False
        self.__state_progress = 0 if self.__report_progress else -1
        self.__state_h = 0 if self.__report_h else -1
        return True

    def parse_dir(self, path: str) -> bool:
        ret: Final[bool] = super().parse_dir(path)
        collect()
        return ret

    def process(self) -> None:
        counter: Final[Counter[int | float]] = self.__counter

        # report goal objective value, if encountered
        if (self.__report_goal_f and (self.goal_f is not None)
                and isfinite(self.goal_f)):
            counter[self.goal_f] = max(counter[self.goal_f], 1)
        # add the known values
        for val in self.__per_instance_known(self.instance):
            if isfinite(val):
                counter[val] = max(counter[val], 1)
        # add the best objective value
        if (self.best_f is not None) and isfinite(self.best_f):
            counter[self.best_f] = max(counter[self.best_f], 1)

        self.__consumer(PerRunData(
            algorithm=self.algorithm,
            instance=self.instance,
            objective=self.objective,
            encoding=self.encoding,
            rand_seed=self.rand_seed), counter)
        counter.clear()

    def start_section(self, title: str) -> bool:
        if title == SECTION_PROGRESS:
            if self.__state_progress >= 2:
                raise ValueError(f"Already did section {title}.")
            if self.__state_progress < 0:
                self.__state_progress = 2
                return False
            self.__state_progress = 1
            return True
        if title == H_LOG_SECTION:
            if self.__state_h >= 2:
                raise ValueError(f"Already did section {title}.")
            if self.__state_h < 0:
                self.__state_h = 2
                return False
            self.__state_h = 1
            return True
        return super().start_section(title)

    def setup_section(self, data: dict[str, str]) -> None:
        """
        Parse the data from the `setup` section.

        :param data: the parsed data
        """
        super().setup_section(data)
        if self.__report_lower_bound and (_FULL_KEY_LOWER_BOUND in data):
            lb: Final[int | float] = str_to_num(
                data[_FULL_KEY_LOWER_BOUND])
            if isfinite(lb):
                self.__counter[lb] = max(self.__counter[lb], 1)
        if self.__report_upper_bound and (_FULL_KEY_UPPER_BOUND in data):
            ub: Final[int | float] = str_to_num(
                data[_FULL_KEY_UPPER_BOUND])
            if isfinite(ub):
                self.__counter[ub] = max(self.__counter[ub], 1)

    def needs_more_lines(self) -> bool:
        return ((self.__state_h == 0) or (self.__state_progress == 0)
                or super().needs_more_lines())

    def lines(self, lines: list[str]) -> bool:
        if not isinstance(lines, list):
            raise type_error(lines, "lines", list)

        counter: Counter[int | float]
        if self.__state_progress == 1:
            self.__state_progress = 2
            columns = [c.strip() for c in lines[0].split(CSV_SEPARATOR)]
            f_col: Final[int] = columns.index(PROGRESS_CURRENT_F)
            counter = self.__counter
            for line in lines[1:]:
                f: int | float = str_to_num(line.split(
                    CSV_SEPARATOR)[f_col])
                counter[f] += 1
        elif self.__state_h == 1:
            self.__state_h = 2
            counter = self.__counter
            for line in lines:
                split = line.split(CSV_SEPARATOR)
                for i in range(0, len(split), 2):
                    counter[str_to_num(split[i])] += int(split[i + 1])
        else:
            return super().lines(lines)

        return self.needs_more_lines()


# Print a CSV file
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipy_argparser(
        __file__, "Collecting the Number of Existing Objective Values.",
        "Gather all the existing objective values and store them in a "
        "CSV-formatted file.")
    parser.add_argument(
        "source", nargs="?", default="./results",
        help="the location of the experimental results, i.e., the root folder "
             "under which to search for log files", type=Path)
    parser.add_argument(
        "dest", help="the path to the end results CSV file to be created",
        type=Path, nargs="?",
        default="./evaluation/objective_values.txt")
    parser.add_argument(
        "--lb", help="count the lower bound of the "
                     "objective as objective value",
        action="store_true")
    parser.add_argument(
        "--ub", help="count the upper bound of the "
                     "objective as objective value",
        action="store_true")
    parser.add_argument(
        "--goal", help="count the goal objective value as "
                       "existing objective value",
        action="store_true")
    args: Final[argparse.Namespace] = parser.parse_args()
    number_of_objective_values_to_csv(
        args.source, args.dest, per_instance=True,
        per_algorithm_instance=True, report_lower_bound=args.lb,
        report_upper_bound=args.ub, report_goal_f=args.goal)
