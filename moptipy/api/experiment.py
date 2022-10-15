"""
The experiment execution API.

Via the function :func:`run_experiment`, you can execute a complex experiment
where multiple optimization algorithms are applied to multiple problem
instances, where log files with the results and progress information about the
runs are collected, and where multiprocessing is used to parallelize the
experiment execution.
Experiments are replicable, as random seeds are automatically generated based
on problem instance names in a replicable fashion.
"""
import copy
import gc
import multiprocessing as mp
import os.path
from contextlib import nullcontext, AbstractContextManager
from math import ceil
from typing import Iterable, Union, Callable, List, Final, Sequence, cast, \
    Any

import psutil  # type: ignore
from numpy.random import Generator, default_rng

from moptipy.api.execution import Execution
from moptipy.api.logging import FILE_SUFFIX
from moptipy.utils.cache import is_new
from moptipy.utils.console import logger
from moptipy.utils.nputils import rand_seeds_from_str
from moptipy.utils.path import Path
from moptipy.utils.strings import sanitize_name, sanitize_names
from moptipy.utils.sys_info import refresh_sys_info
from moptipy.utils.types import type_error


def __run_experiment(base_dir: Path,
                     experiments: List[List[Callable]],
                     n_runs: List[int],
                     perform_warmup: bool,
                     warmup_fes: int,
                     perform_pre_warmup: bool,
                     pre_warmup_fes: int,
                     file_lock: AbstractContextManager,
                     stdio_lock: AbstractContextManager,
                     cache: Callable[[str], bool],
                     thread_id: str,
                     pre_warmup_barrier) -> None:
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
                seeds: List[int] = [0] if warmup else \
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
                            f"{wss} for '{filename}'.", thread_id, stdio_lock)
                        with cpy.execute():
                            pass
                        del cpy

                    if warmup:
                        continue

                    exp.set_log_file(log_file)
                    logger(filename, thread_id, stdio_lock)
                    with exp.execute():  # run the experiment
                        pass


#: the number of logical CPU cores
__CPU_LOGICAL_CORES: Final[int] = psutil.cpu_count(logical=True)
#: the number of phyiscal CPU cores
__CPU_PHYSICAL_CORES: Final[int] = psutil.cpu_count(logical=False)
#: the logical cores per physical core
__CPU_LOGICAL_PER_PHYSICAL: Final[int] = \
    max(1, int(ceil(__CPU_LOGICAL_CORES / __CPU_PHYSICAL_CORES)))
#: The default number of threads to be used
__DEFAULT_N_THREADS: Final[int] = max(1, __CPU_PHYSICAL_CORES - 1)


def __waiting_run_experiment(base_dir: Path,
                             experiments: List[List[Callable]],
                             n_runs: List[int],
                             perform_warmup: bool,
                             warmup_fes: int,
                             perform_pre_warmup: bool,
                             pre_warmup_fes: int,
                             file_lock: AbstractContextManager,
                             stdio_lock: AbstractContextManager,
                             cache: Callable,
                             thread_id: str,
                             event, pre_warmup_barrier) -> None:
    """Wait until event is set, then run experiment."""
    logger("waiting for start signal", thread_id, stdio_lock)
    if not event.wait():
        raise ValueError("Wait terminated unexpectedly.")
    logger("got start signal, beginning experiment", thread_id, stdio_lock)
    refresh_sys_info()
    gc.collect()
    __run_experiment(base_dir, experiments, n_runs,
                     perform_warmup, warmup_fes, perform_pre_warmup,
                     pre_warmup_fes, file_lock, stdio_lock, cache,
                     thread_id, pre_warmup_barrier)


def run_experiment(base_dir: str,
                   instances: Iterable[Callable[[], Any]],
                   setups: Iterable[Callable[[Any], Execution]],
                   n_runs: Union[int, Iterable[int]] = 11,
                   n_threads: int = __DEFAULT_N_THREADS,
                   perform_warmup: bool = True,
                   warmup_fes: int = 20,
                   perform_pre_warmup: bool = True,
                   pre_warmup_fes: int = 20) -> Path:
    """
    Run an experiment and store the log files into the given folder.

    This function will automatically run an experiment, i.e., apply a set
    `setups` of algorithm setups to a set `instances` of problem instances for
    `n_runs` each. It will collect log files and store them into an
    appropriate folder structure under the path `base_dir`. It will use
    `n_threads` separate processes to parallelize the whole experiment (if you
    do not specify `n_threads`, it will be chosen automatically).

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
        By default, we will use the number of physical cores - 1 processes.
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
    if not isinstance(warmup_fes, int):
        raise type_error(warmup_fes, "warmup_fes", int)
    if warmup_fes <= 0:
        raise ValueError(f"warmup_fes must > 0, but is {warmup_fes}.")
    if not isinstance(pre_warmup_fes, int):
        raise type_error(pre_warmup_fes, "pre_warmup_fes", int)
    if pre_warmup_fes <= 0:
        raise ValueError(f"warmup_fes must > 0, but is {pre_warmup_fes}.")
    if not isinstance(n_threads, int):
        raise type_error(n_threads, "n_threads", int)
    if n_threads <= 0:
        raise ValueError(f"n_threads must be positive, but is {n_threads}.")

    instances = list(instances)
    if len(instances) <= 0:
        raise ValueError("Instance enumeration is empty.")
    for instance in instances:
        if not callable(instance):
            raise type_error(instance, "all instances", call=True)

    setups = list(setups)
    if len(setups) <= 0:
        raise ValueError("Setup enumeration is empty.")
    for setup in setups:
        if not callable(setup):
            raise type_error(setup, "all setups", call=True)

    experiments: Final[List[List[Callable]]] = \
        [[ii, ss] for ii in instances for ss in setups]

    del instances
    del setups

    if len(experiments) <= 0:
        raise ValueError("No experiments found?")

    if isinstance(n_runs, int):
        n_runs = [n_runs, ]
    else:
        n_runs = list(n_runs)
    last = 0
    for run in n_runs:
        if not isinstance(run, int):
            raise type_error(run, "n_runs", int)
        if run <= last:
            raise ValueError(
                "n_runs sequence must be strictly increasing and "
                f"positive, we cannot have {run} follow {last}.")
        last = run

    cache: Final[Callable[[str], bool]] = is_new()
    use_dir: Final[Path] = Path.path(base_dir)
    use_dir.ensure_dir_exists()

    stdio_lock: AbstractContextManager

    if n_threads > 1:
        file_lock: AbstractContextManager = mp.Lock()
        stdio_lock = mp.Lock()
        logger(f"starting experiment with {n_threads} threads "
               f"on {__CPU_LOGICAL_CORES} logical cores, "
               f"{__CPU_PHYSICAL_CORES} physical cores (i.e.,"
               f" {__CPU_LOGICAL_PER_PHYSICAL} logical cores per physical "
               "core).", "", stdio_lock)

        event: Final = mp.Event()
        pre_warmup_barrier: Final = mp.Barrier(n_threads) \
            if perform_pre_warmup else None
        processes: Final[List[mp.Process]] = \
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
                              pre_warmup_barrier))
             for i in range(n_threads)]

        for i, p in enumerate(processes):
            p.start()
            logger(f"started processes {hex(i)[2:]} in waiting state.",
                   "", stdio_lock)

        # try to distribute the load evenly over all cores
        n_cpus: int = __CPU_PHYSICAL_CORES
        core_ofs: int = 0
        if n_threads < n_cpus:
            n_cpus -= 1
            core_ofs = __CPU_LOGICAL_PER_PHYSICAL
        n_cores: Final[int] = n_cpus * __CPU_LOGICAL_PER_PHYSICAL
        n_cores_per_thread: Final[int] = max(1, n_cores // n_threads)

        last_core: int = 0
        for i, p in enumerate(processes):
            pid: int = int(p.pid)
            aff: List[int] = []
            for _ in range(n_cores_per_thread):
                aff.append(int((last_core + core_ofs) % __CPU_LOGICAL_CORES))
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
               f"on {__CPU_LOGICAL_CORES} logical cores, "
               f"{__CPU_PHYSICAL_CORES} physical cores (i.e.,"
               f" {__CPU_LOGICAL_PER_PHYSICAL} logical cores per physical "
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
                         pre_warmup_barrier=None)

    logger("finished experiment.", "", stdio_lock)
    return use_dir
