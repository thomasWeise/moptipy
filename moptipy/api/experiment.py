"""
The experiment execution API.

Via the function :func:`run_experiment`, you can execute a complex experiment
where multiple optimization algorithms are applied to multiple problem
instances, where log files with the results and progress information about the
runs are collected, and where multiprocessing is used to parallelize the
experiment execution.
Experiments are replicable, as random seeds are automatically generated based
on problem instance names in a replicable fashion.

The log files are structured according to the documentation in
https://thomasweise.github.io/moptipy/#file-names-and-folder-structure
and their contents follow the specification given in
https://thomasweise.github.io/moptipy/#log-file-sections.
"""
import copy
import gc
import os.path
from os import getpid
from typing import Any, Callable, Final, Iterable, Sequence, cast

from numpy.random import Generator, default_rng
from pycommons.ds.cache import str_is_new
from pycommons.io.console import logger
from pycommons.io.path import Path
from pycommons.types import check_int_range, type_error

from moptipy.api.execution import Execution
from moptipy.api.logging import FILE_SUFFIX
from moptipy.api.process import Process
from moptipy.utils.nputils import rand_seeds_from_str
from moptipy.utils.strings import sanitize_name, sanitize_names
from moptipy.utils.sys_info import get_sys_info


def __run_experiment(base_dir: Path,
                     experiments: list[list[Callable]],
                     n_runs: list[int],
                     thread_id: str,
                     perform_warmup: bool,
                     warmup_fes: int,
                     perform_pre_warmup: bool,
                     pre_warmup_fes: int,
                     on_completion: Callable[[
                         Any, Path, Process], None]) -> None:
    """
    Execute a single thread of experiments.

    :param base_dir: the base directory
    :param experiments: the stream of experiment setups
    :param n_runs: the list of runs
    :param thread_id: the thread id
    :param perform_warmup: should we perform a warm-up per instance?
    :param warmup_fes: the number of the FEs for the warm-up runs
    :param perform_pre_warmup: should we do one warmup run for each
        instance before we begin with the actual experiments?
    :param pre_warmup_fes: the FEs for the pre-warmup runs
    :param on_completion: a function to be called for every completed run,
        receiving the instance, the path to the log file (before it is
        created) and the :class:`~moptipy.api.process.Process` of the run
        as parameters
    """
    random: Final[Generator] = default_rng()
    cache: Final[Callable[[str], bool]] = str_is_new()
    for warmup in ([True, False] if perform_pre_warmup else [False]):
        wss: str
        if warmup:
            wss = "pre-warmup"
        else:
            wss = "warmup"
            if perform_pre_warmup:
                gc.collect()  # do full garbage collection after pre-warmups
                gc.collect()  # one more, to be double-safe
                gc.freeze()  # whatever survived now, keep it permanently

        for runs in ([1] if warmup else n_runs):  # for each number of runs
            if not warmup:
                logger(f"now doing {runs} runs.", thread_id)
            random.shuffle(cast(Sequence, experiments))  # shuffle experiments

            for setup in experiments:  # for each setup
                instance = setup[0]()  # load instance
                if instance is None:
                    raise TypeError("None is not an instance.")
                inst_name = sanitize_name(str(instance))

                exp = setup[1](instance)  # setup algorithm for instance
                if not isinstance(exp, Execution):
                    raise type_error(exp, "result of setup callable",
                                     Execution)
                # noinspection PyProtectedMember
                algo_name = sanitize_name(str(exp._algorithm))

                cd = Path(os.path.join(base_dir, algo_name, inst_name))
                cd.ensure_dir_exists()

                # generate sequence of seeds
                seeds: list[int] = [0] if warmup else \
                    rand_seeds_from_str(string=inst_name, n_seeds=runs)
                random.shuffle(seeds)
                needs_warmup = warmup or perform_warmup
                for seed in seeds:  # for every run

                    filename = sanitize_names(
                        [algo_name, inst_name, hex(seed)])
                    if warmup:
                        log_file = filename
                    else:
                        log_file = Path(
                            os.path.join(cd, filename + FILE_SUFFIX))

                        skip = True
                        if cache(log_file):
                            skip = log_file.ensure_file_exists()
                        if skip:
                            continue  # run already done

                    exp.set_rand_seed(seed)

                    if needs_warmup:  # perform warmup run
                        needs_warmup = False
                        cpy: Execution = copy.copy(exp)
                        cpy.set_max_fes(
                            pre_warmup_fes if warmup else warmup_fes, True)
                        cpy.set_max_time_millis(3600000, True)
                        cpy.set_log_file(None)
                        cpy.set_log_improvements(False)
                        cpy.set_log_all_fes(False)
                        logger(f"{wss} for {filename!r}.", thread_id)
                        with cpy.execute():
                            pass
                        del cpy

                    if warmup:
                        continue

                    exp.set_log_file(log_file)
                    logger(filename, thread_id)
                    with exp.execute() as process:  # run the experiment
                        on_completion(instance, cast(Path, log_file), process)


def __no_complete(_: Any, __: Path, ___: Process) -> None:
    """Do nothing."""


def run_experiment(
        base_dir: str, instances: Iterable[Callable[[], Any]],
        setups: Iterable[Callable[[Any], Execution]],
        n_runs: int | Iterable[int] = 11,
        perform_warmup: bool = True, warmup_fes: int = 20,
        perform_pre_warmup: bool = True, pre_warmup_fes: int = 20,
        on_completion: Callable[[Any, Path, Process], None] = __no_complete) \
        -> Path:
    """
    Run an experiment and store the log files into the given folder.

    This function will automatically run an experiment, i.e., apply a set
    `setups` of algorithm setups to a set `instances` of problem instances for
    `n_runs` each. It will collect log files and store them into an
    appropriate folder structure under the path `base_dir`. It will
    automatically draw random seeds for all algorithm runs using
    :func:`moptipy.utils.nputils.rand_seeds_from_str` based on the names of
    the problem instances to solve. This yields replicable experiments, i.e.,
    running the experiment program twice will yield exactly the same runs in
    exactly the same file structure (give and take clock-time dependent
    issues, which obviously cannot be controlled in a deterministic fashion).

    :param base_dir: the base directory where to store the results
    :param instances: an iterable of callables, each of which should return an
        object representing a problem instance, whose `__str__` representation
        is a valid name
    :param setups: an iterable of callables, each receiving an instance (as
        returned by instances) as input and producing an
        :class:`moptipy.api.execution.Execution` as output
    :param n_runs: the number of runs per algorithm-instance combination
    :param perform_warmup: should we perform a warm-up for each instance?
        If this parameter is `True`, then before the very first run of a
        thread on an instance, we will execute the algorithm for just a few
        function evaluations without logging and discard the results. The
        idea is that during this warm-up, things such as JIT compilation or
        complicated parsing can take place. While this cannot mitigate time
        measurement problems for JIT compilations taking place late in runs,
        it can at least somewhat solve the problem of delayed first FEs caused
        by compilation and parsing.
    :param warmup_fes: the number of the FEs for the warm-up runs
    :param perform_pre_warmup: should we do one warmup run for each
        instance before we begin with the actual experiments? This complements
        the warmups defined by `perform_warmup`. It could be that, for some
        reason, JIT or other activities may lead to stalls between multiple
        processes when code is encountered for the first time. This may or may
        not still cause strange timing issues even if `perform_warmup=True`.
        We therefore can do one complete round of warmups before starting the
        actual experiment. After that, we perform one garbage collection run
        and then freeze all objects surviving it to prevent them from future
        garbage collection runs. All processes that execute the experiment in
        parallel will complete their pre-warmup and only after all of them have
        completed it, the actual experiment will begin. I am not sure whether
        this makes sense or not, but it also would not hurt.
    :param pre_warmup_fes: the FEs for the pre-warmup runs
    :param on_completion: a function to be called for every completed run,
        receiving the instance, the path to the log file (before it is
        created) and the :class:`~moptipy.api.process.Process` of the run
        as parameters

    :returns: the canonicalized path to `base_dir`
    """
    if not isinstance(instances, Iterable):
        raise type_error(instances, "instances", Iterable)
    if not isinstance(setups, Iterable):
        raise type_error(setups, "setups", Iterable)
    if not isinstance(perform_warmup, bool):
        raise type_error(perform_warmup, "perform_warmup", bool)
    if not isinstance(perform_pre_warmup, bool):
        raise type_error(perform_pre_warmup, "perform_pre_warmup", bool)
    check_int_range(warmup_fes, "warmup_fes", 1, 1_000_000)
    check_int_range(pre_warmup_fes, "pre_warmup_fes", 1, 1_000_000)

    instances = list(instances)
    if list.__len__(instances) <= 0:
        raise ValueError("Instance enumeration is empty.")
    for instance in instances:
        if not callable(instance):
            raise type_error(instance, "all instances", call=True)

    if str.__len__(get_sys_info()) <= 0:
        raise ValueError("empty system info?")

    setups = list(setups)
    if list.__len__(setups) <= 0:
        raise ValueError("Setup enumeration is empty.")
    for setup in setups:
        if not callable(setup):
            raise type_error(setup, "all setups", call=True)

    experiments: Final[list[list[Callable]]] = \
        [[ii, ss] for ii in instances for ss in setups]

    del instances
    del setups

    if list.__len__(experiments) <= 0:
        raise ValueError("No experiments found?")

    n_runs = [n_runs] if isinstance(n_runs, int) else list(n_runs)
    if list.__len__(n_runs) <= 0:
        raise ValueError("No number of runs provided?")
    last = 0
    for run in n_runs:
        last = check_int_range(run, "n_runs", last + 1)

    use_dir: Final[Path] = Path(base_dir)
    use_dir.ensure_dir_exists()

    thread_id: Final[str] = f"@{getpid():x}"
    logger("beginning experiment execution.", thread_id)
    __run_experiment(base_dir=use_dir,
                     experiments=experiments,
                     n_runs=n_runs,
                     thread_id=thread_id,
                     perform_warmup=perform_warmup,
                     warmup_fes=warmup_fes,
                     perform_pre_warmup=perform_pre_warmup,
                     pre_warmup_fes=pre_warmup_fes,
                     on_completion=on_completion)
    logger("finished experiment execution.", thread_id)
    return use_dir
