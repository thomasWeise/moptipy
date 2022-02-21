"""The experiment execution API."""
import copy
import gc
import multiprocessing as mp
import os.path
from contextlib import nullcontext
from math import ceil
from typing import Iterable, Union, Callable, Tuple, List, \
    ContextManager, Final, Sequence, cast

import psutil  # type: ignore
from numpy.random import Generator, default_rng

from moptipy.api.execution import Execution
from moptipy.api.logging import sanitize_name, sanitize_names, FILE_SUFFIX
from moptipy.utils.cache import is_new
from moptipy.utils.log import logger
from moptipy.utils.nputils import rand_seeds_from_str
from moptipy.utils.path import Path
from moptipy.utils.sys_info import refresh_sys_info


def __run_experiment(base_dir: Path,
                     experiments: List[Tuple[Callable, Callable]],
                     n_runs: Tuple[int, ...],
                     perform_warmup: bool,
                     warmup_fes: int,
                     perform_pre_warmup: bool,
                     pre_warmup_fes: int,
                     file_lock: ContextManager,
                     stdio_lock: ContextManager,
                     cache: Callable,
                     thread_id: str,
                     pre_warmup_barrier) -> None:
    """
    Execute a single thread of experiments.

    :param str base_dir: the base directory
    :param List[Tuple[Callable, Callable]] experiments: the stream of
        experiment setups
    :param bool perform_warmup: should we perform a warm-up per instance?
    :param int warmup_fes: the number of the FEs for the warm-up runs
    :param bool perform_pre_warmup: should we do one warmup run for each
        instance before we begin with the actual experiments?
    :param int pre_warmup_fes: the FEs for the pre-warmup runs
    :param ContextManager file_lock: the lock for file operations
    :param ContextManager stdio_lock: the lock for log output
    :param Callable cache: the cache
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
                    raise ValueError("None is not an instance.")
                inst_name = sanitize_name(str(instance))

                exp = setup[1](instance)  # setup algorithm for instance
                if not isinstance(exp, Execution):
                    raise ValueError(
                        "Setup callable must produce instance of "
                        f"Execution, but generates {type(exp)}.")
                algo_name = sanitize_name(exp.get_algorithm().get_name())

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
                             experiments: List[Tuple[Callable, Callable]],
                             n_runs: Tuple[int, ...],
                             perform_warmup: bool,
                             warmup_fes: int,
                             perform_pre_warmup: bool,
                             pre_warmup_fes: int,
                             file_lock: ContextManager,
                             stdio_lock: ContextManager,
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
                   instances: Iterable[Callable],
                   setups: Iterable[Callable],
                   n_runs: Union[int, Iterable[int]] = 11,
                   n_threads: int = __DEFAULT_N_THREADS,
                   perform_warmup: bool = True,
                   warmup_fes: int = 20,
                   perform_pre_warmup: bool = True,
                   pre_warmup_fes: int = 20) -> Path:
    """
    Run an experiment and store the log files into the given folder.

    :param str base_dir: the base directory where to store the results
    :param Iterable[Callable] instances: an iterable of callables, each of
        which should return an object representing a problem instance, whose
        :func:`str` representation is a valid name
    :param Iterable[Callable] setups: an iterable of callables, each receiving
        an instance (as returned by instances) as input and producing an
        :class:`Execution` as output
    :param Union[int, Iterable[int]] n_runs: the number of runs per algorithm-
        instance combination
    :param int n_threads: the number of parallel threads of execution to use.
        By default, we will use the number of physical cores - 1 processes.
        We will try to distribute the threads over different logical and
        physical cores to minimize their interactions. If n_threads is less
        or equal the number of physical cores, then multiple logical cores
        will be assigned to each process.
        If less threads than the number of physical cores are spawned, we will
        leave one physical core unoccupied. This core may be used by the
        operating system or other processes for their work, thus reducing
        interference of the os with our experiments.
    :param bool perform_warmup: should we perform a warm-up for each instance?
        If this parameter is `True`, then before the very first run of a
        thread on an instance, we will execute the algorithm for just a few
        function evaluations without logging and discard the results. The
        idea is that during this warm-up, things such as JIT compilation or
        complicated parsing can take place. While this cannot mitigate time
        measurement problems for JIT compilations taking place late in runs,
        it can at least somewhat solve the problem of delayed first FEs caused
        by compilation and parsing.
    :param int warmup_fes: the number of the FEs for the warm-up runs
    :param bool perform_pre_warmup: should we do one warmup run for each
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
    :param int pre_warmup_fes: the FEs for the pre-warmup runs

    :returns: the canonicalized path to `base_dir`
    :rtype: Path
    """
    if not isinstance(instances, Iterable):
        raise TypeError(
            f"instances must be a iterable object, but is {type(instances)}.")
    if not isinstance(setups, Iterable):
        raise TypeError(
            f"setups must be a iterable object, but is {type(setups)}.")
    if not isinstance(perform_warmup, bool):
        raise TypeError(
            f"perform_warmup must be a bool, but is {type(perform_warmup)}.")
    if not isinstance(perform_pre_warmup, bool):
        raise TypeError(f"perform_pre_warmup must be a bool, "
                        f"but is {type(perform_pre_warmup)}.")
    if not isinstance(warmup_fes, int):
        raise TypeError(f"warmup_fes must be int, but is {type(warmup_fes)}.")
    if warmup_fes <= 0:
        raise TypeError(f"warmup_fes must > 0, but is {warmup_fes}.")
    if not isinstance(pre_warmup_fes, int):
        raise TypeError(
            f"pre_warmup_fes must be int, but is {type(pre_warmup_fes)}.")
    if pre_warmup_fes <= 0:
        raise TypeError(f"warmup_fes must > 0, but is {pre_warmup_fes}.")

    if not isinstance(n_threads, int):
        raise TypeError(f"n_threads must be int, but is {type(n_threads)}.")
    if n_threads <= 0:
        raise ValueError(f"n_threads must be positive, but is {n_threads}.")

    instances = list(instances)
    if len(instances) <= 0:
        raise ValueError("Instance enumeration is empty.")
    for instance in instances:
        if not callable(instance):
            raise TypeError("All instances must be callables, "
                            f"but encountered a {type(instance)}.")

    setups = list(setups)
    if len(setups) <= 0:
        raise ValueError("Setup enumeration is empty.")
    for setup in setups:
        if not callable(setup):
            raise TypeError("All setups must be callables, "
                            f"but encountered a {type(setup)}.")

    experiments: Final[List[Tuple[Callable, Callable]]] = \
        [(ii, ss) for ii in instances for ss in setups]

    del instances
    del setups

    if len(experiments) <= 0:
        raise ValueError("No experiments found?")

    if isinstance(n_runs, int):
        n_runs = (n_runs,)
    else:
        n_runs = tuple(n_runs)
    last = 0
    for run in n_runs:
        if run is None:
            raise ValueError("n_runs must not be None.")
        if run <= last:
            raise ValueError("n_runs sequence must be increasing and "
                             f"positive, we cannot have {run} follow {last}.")
        last = run

    cache = is_new()
    use_dir: Final[Path] = Path.path(base_dir)
    use_dir.ensure_dir_exists()

    stdio_lock: ContextManager

    if n_threads > 1:
        file_lock: ContextManager = mp.Lock()
        stdio_lock = mp.Lock()
        logger(f"starting experiment with {n_threads} threads "
               f"on {__CPU_LOGICAL_CORES} logical cores, "
               f"{__CPU_PHYSICAL_CORES} physical cores (i.e.,"
               f" {__CPU_LOGICAL_PER_PHYSICAL} logical cores per physical "
               "core).", "", stdio_lock)

        event: Final = mp.Event()
        pre_warmup_barrier: Final = mp.Barrier(n_threads) \
            if perform_pre_warmup else None
        processes = [mp.Process(target=__waiting_run_experiment,
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
