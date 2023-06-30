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
import multiprocessing as mp
import os.path
import platform
from contextlib import AbstractContextManager, nullcontext
from enum import IntEnum
from math import ceil
from typing import Any, Callable, Final, Iterable, Sequence, cast

import psutil  # type: ignore
from numpy.random import Generator, default_rng

from moptipy.api.execution import Execution
from moptipy.api.logging import FILE_SUFFIX
from moptipy.api.process import Process
from moptipy.utils.cache import is_new
from moptipy.utils.console import logger
from moptipy.utils.nputils import rand_seeds_from_str
from moptipy.utils.path import Path
from moptipy.utils.strings import sanitize_name, sanitize_names
from moptipy.utils.sys_info import get_sys_info, update_sys_info_cpu_affinity
from moptipy.utils.types import check_int_range, type_error


def __run_experiment(base_dir: Path,
                     experiments: list[list[Callable]],
                     n_runs: list[int],
                     perform_warmup: bool,
                     warmup_fes: int,
                     perform_pre_warmup: bool,
                     pre_warmup_fes: int,
                     file_lock: AbstractContextManager,
                     stdio_lock: AbstractContextManager,
                     cache: Callable[[str], bool],
                     thread_id: str,
                     pre_warmup_barrier,
                     on_completion: Callable[[
                         Any, Path, Process], None]) -> None:
    """
    Execute a single thread of experiments.

    :param base_dir: the base directory
    :param experiments: the stream of experiment setups
    :param perform_warmup: should we perform a warm-up per instance?
    :param warmup_fes: the number of the FEs for the warm-up runs
    :param perform_pre_warmup: should we do one warmup run for each
        instance before we begin with the actual experiments?
    :param pre_warmup_fes: the FEs for the pre-warmup runs
    :param file_lock: the lock for file operations
    :param stdio_lock: the lock for log output
    :param cache: the cache
    :param thread_id: the thread id
    :param pre_warmup_barrier: a barrier to wait at after the pre-warmup
    :param on_completion: a function to be called for every completed run,
        receiving the instance, the path to the log file (before it is
        created) and the :class:`~moptipy.api.process.Process` of the run
        as parameters
    """
    random: Final[Generator] = default_rng()

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
                if pre_warmup_barrier:
                    logger(
                        "reached pre-warmup barrier.", thread_id, stdio_lock)
                    pre_warmup_barrier.wait()  # wait for all threads

        for runs in ([1] if warmup else n_runs):  # for each number of runs
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

                cd = Path.path(os.path.join(base_dir, algo_name, inst_name))
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
                        log_file = Path.path(
                            os.path.join(cd, filename + FILE_SUFFIX))

                        skip = True
                        with file_lock:
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
                        logger(
                            f"{wss} for {filename!r}.", thread_id, stdio_lock)
                        with cpy.execute():
                            pass
                        del cpy

                    if warmup:
                        continue

                    exp.set_log_file(log_file)
                    logger(filename, thread_id, stdio_lock)
                    with exp.execute() as process:  # run the experiment
                        on_completion(instance, cast(Path, log_file), process)


#: the number of logical CPU cores
_CPU_LOGICAL_CORES: Final[int] = psutil.cpu_count(logical=True)
#: the number of phyiscal CPU cores
_CPU_PHYSICAL_CORES: Final[int] = psutil.cpu_count(logical=False)
#: the logical cores per physical core
_CPU_LOGICAL_PER_PHYSICAL: Final[int] = \
    max(1, int(ceil(_CPU_LOGICAL_CORES / _CPU_PHYSICAL_CORES)))


class Parallelism(IntEnum):
    """
    An enumeration of parallel thread counts.

    The `value` of each element of this enumeration equal the number of
    threads that would be used. Thus, the values are different on
    different systems. Currently, only Linux is supported for parallelism.
    For all other systems, you need to manually start the program as often
    as you want it to run in parallel. On other systems, all values of this
    enumeration are `1`.

    >>> Parallelism.SINGLE_THREAD.value
    1
    """

    #: use only a single thread
    SINGLE_THREAD = 1
    #: Use as many threads as accurate time measurement permits. This equals
    #: to using one logical core on each physical core and leaving one
    #: physical core unoccupied (but always using at least one thread,
    #: obviously). If you have four physical cores with two logical cores
    #: each, this would mean using three threads on Linux.
    #: On Windows, parallelism is not supported yet, so this would equal using
    #: only one core on Windows.
    ACCURATE_TIME_MEASUREMENTS = max(1, _CPU_PHYSICAL_CORES - 1) \
        if "Linux" in platform.system() else 1
    #: Use all but one logical core. This *will* mess up time measurements but
    #: should produce the maximum performance while not impeding the system of
    #: doing stuff like garbage collection or other bookkeeping and overhead
    #: tasks. This is the most reasonable option if you want to execute one
    #: experiment as quickly as possible. If you have four physical cores with
    #: two logical cores each, this would mean using seven threads on Linux.
    #: On Windows, parallelism is not supported yet, so this would equal using
    #: only one core on Windows.
    PERFORMANCE = max(1, _CPU_LOGICAL_CORES - 1) \
        if "Linux" in platform.system() else 1
    #: Use every single logical core available, which may mess up your system.
    #: We run the experiment as quickly as possible, but the system may not be
    #: usable while the experiment is running. Background tasks like garbage
    #: collection, moving the mouse course, accepting user input, or network
    #: communication may come to a halt. Seriously, why would you use this?
    #: If you have four physical cores with  two logical cores each, this would
    #: mean using eight threads on Linux.
    #: On Windows, parallelism is not supported yet, so this would equal using
    #: only one core on Windows.
    RECKLESS = max(1, _CPU_LOGICAL_CORES) \
        if "Linux" in platform.system() else 1


def __waiting_run_experiment(
        base_dir: Path, experiments: list[list[Callable]],
        n_runs: list[int], perform_warmup: bool, warmup_fes: int,
        perform_pre_warmup: bool, pre_warmup_fes: int,
        file_lock: AbstractContextManager,
        stdio_lock: AbstractContextManager, cache: Callable, thread_id: str,
        event, pre_warmup_barrier,
        on_completion: Callable[[Any, Path, Process], None]) -> None:
    """Wait until event is set, then run experiment."""
    logger("waiting for start signal", thread_id, stdio_lock)
    if not event.wait():
        raise ValueError("Wait terminated unexpectedly.")
    logger("got start signal, beginning experiment", thread_id, stdio_lock)
    update_sys_info_cpu_affinity()
    gc.collect()
    __run_experiment(base_dir, experiments, n_runs,
                     perform_warmup, warmup_fes, perform_pre_warmup,
                     pre_warmup_fes, file_lock, stdio_lock, cache,
                     thread_id, pre_warmup_barrier, on_completion)


def __no_complete(_: Any, __: Path, ___: Process) -> None:
    """Do nothing."""


def run_experiment(
        base_dir: str, instances: Iterable[Callable[[], Any]],
        setups: Iterable[Callable[[Any], Execution]],
        n_runs: int | Iterable[int] = 11,
        n_threads: int = Parallelism.ACCURATE_TIME_MEASUREMENTS,
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

    This function will use `n_threads` separate processes to parallelize the
    whole experiment (if you do not specify `n_threads`, it will be chosen
    automatically).

    Note for Windows users: The parallelization will not work under Windows.
    However, you can achieve *almost* the same effect and performance as for
    `n_threads=N` if you set `n_threads=1` and simply start the program `N`
    times separately (in separate terminals and in parallel). Of course, all
    `N` processes must have the same `base_dir` parameter. They will then
    automatically share the workload.

    :param base_dir: the base directory where to store the results
    :param instances: an iterable of callables, each of which should return an
        object representing a problem instance, whose `__str__` representation
        is a valid name
    :param setups: an iterable of callables, each receiving an instance (as
        returned by instances) as input and producing an
        :class:`moptipy.api.execution.Execution` as output
    :param n_runs: the number of runs per algorithm-instance combination
    :param n_threads: the number of parallel threads of execution to use.
        This parameter only works under Linux! It should be set to 1 under all
        other operating systems. Under Linux, by default, we will use the
        number of physical cores - 1 processes.
        The default value for `n_threads` is computed in \
:py:const:`~moptipy.api.experiment.Parallelism.ACCURATE_TIME_MEASUREMENTS`,
        which will be different for different machines(!).
        We will try to distribute the threads over different logical and
        physical cores to minimize their interactions. If n_threads is less
        or equal the number of physical cores, then multiple logical cores
        will be assigned to each process.
        If less threads than the number of physical cores are spawned, we will
        leave one physical core unoccupied. This core may be used by the
        operating system or other processes for their work, thus reducing
        interference of the os with our experiments.
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
    check_int_range(n_threads, "n_threads", 1, 16384)
    instances = list(instances)
    if len(instances) <= 0:
        raise ValueError("Instance enumeration is empty.")
    for instance in instances:
        if not callable(instance):
            raise type_error(instance, "all instances", call=True)

    sysinfo_check: str = get_sys_info()
    if not isinstance(sysinfo_check, str):
        raise type_error(sysinfo_check, "system information", str)
    if len(sysinfo_check) <= 0:
        raise ValueError(f"invalid system info {sysinfo_check!r}!")

    setups = list(setups)
    if len(setups) <= 0:
        raise ValueError("Setup enumeration is empty.")
    for setup in setups:
        if not callable(setup):
            raise type_error(setup, "all setups", call=True)

    experiments: Final[list[list[Callable]]] = \
        [[ii, ss] for ii in instances for ss in setups]

    del instances
    del setups

    if len(experiments) <= 0:
        raise ValueError("No experiments found?")

    n_runs = [n_runs] if isinstance(n_runs, int) else list(n_runs)
    last = 0
    for run in n_runs:
        last = check_int_range(run, "n_runs", last + 1)

    cache: Final[Callable[[str], bool]] = is_new()
    use_dir: Final[Path] = Path.path(base_dir)
    use_dir.ensure_dir_exists()

    stdio_lock: AbstractContextManager

    if n_threads > 1:
        file_lock: AbstractContextManager = mp.Lock()
        stdio_lock = mp.Lock()
        logger(f"starting experiment with {n_threads} threads "
               f"on {_CPU_LOGICAL_CORES} logical cores, "
               f"{_CPU_PHYSICAL_CORES} physical cores (i.e.,"
               f" {_CPU_LOGICAL_PER_PHYSICAL} logical cores per physical "
               "core).", "", stdio_lock)

        event: Final = mp.Event()
        pre_warmup_barrier: Final = mp.Barrier(n_threads) \
            if perform_pre_warmup else None
        processes: Final[list[mp.Process]] = \
            [mp.Process(target=__waiting_run_experiment,
                        args=(use_dir,
                              experiments.copy(),
                              n_runs,
                              perform_warmup,
                              warmup_fes,
                              perform_pre_warmup,
                              pre_warmup_fes,
                              file_lock,
                              stdio_lock,
                              cache,
                              ":" + hex(i)[2:],
                              event,
                              pre_warmup_barrier,
                              on_completion))
             for i in range(n_threads)]

        for i, p in enumerate(processes):
            p.start()
            logger(f"started processes {hex(i)[2:]} in waiting state.",
                   "", stdio_lock)

        # try to distribute the load evenly over all cores
        n_cpus: int = _CPU_PHYSICAL_CORES
        core_ofs: int = 0
        if n_threads < n_cpus:
            n_cpus -= 1
            core_ofs = _CPU_LOGICAL_PER_PHYSICAL
        n_cores: Final[int] = n_cpus * _CPU_LOGICAL_PER_PHYSICAL
        n_cores_per_thread: Final[int] = max(1, n_cores // n_threads)

        last_core: int = 0
        for i, p in enumerate(processes):
            pid: int = int(p.pid)
            aff: list[int] = []
            for _ in range(n_cores_per_thread):
                aff.append(int((last_core + core_ofs) % _CPU_LOGICAL_CORES))
                last_core += 1
            psutil.Process(pid).cpu_affinity(aff)
            logger(f"set affinity of processes {hex(i)[2:]} with "
                   f"pid {pid} ({hex(pid)}) to {aff}.", "", stdio_lock)
        logger("now releasing lock and starting all processes.",
               "", stdio_lock)
        event.set()
        for i, p in enumerate(processes):
            p.join()
            logger(f"processes {hex(i)[2:]} has finished.", "", stdio_lock)

    else:
        logger(f"starting experiment with single thread "
               f"on {_CPU_LOGICAL_CORES} logical cores, "
               f"{_CPU_PHYSICAL_CORES} physical cores (i.e.,"
               f" {_CPU_LOGICAL_PER_PHYSICAL} logical cores per physical "
               "core).")
        stdio_lock = nullcontext()
        __run_experiment(base_dir=use_dir,
                         experiments=experiments,
                         n_runs=n_runs,
                         perform_warmup=perform_warmup,
                         warmup_fes=warmup_fes,
                         perform_pre_warmup=perform_pre_warmup,
                         pre_warmup_fes=pre_warmup_fes,
                         file_lock=nullcontext(),
                         stdio_lock=stdio_lock,
                         cache=cache,
                         thread_id="",
                         pre_warmup_barrier=None,
                         on_completion=on_completion)

    logger("finished experiment.", "", stdio_lock)
    return use_dir
