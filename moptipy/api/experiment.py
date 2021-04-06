"""The experiment execution API."""
import copy
import multiprocessing as mp
import os.path
from datetime import datetime
from os import makedirs, sched_getaffinity
from typing import Iterable, Union, Callable, Tuple, List, \
    ContextManager, Final

from numpy.random import default_rng

from moptipy.api.execution import Execution
from moptipy.utils.cache import is_new
from moptipy.utils.io import canonicalize_path, enforce_dir
from moptipy.utils.io import file_ensure_exists
from moptipy.utils.logging import sanitize_name, sanitize_names, FILE_SUFFIX
from moptipy.utils.nputils import rand_seeds_from_str


def __ensure_dir(dir_name: str) -> str:
    """
    Make sure that the directory referenced by `dir_name` exists.

    :param str dir_name: the directory name
    :return: the string
    :rtype: str
    """
    dir_name = canonicalize_path(dir_name)
    makedirs(name=dir_name, exist_ok=True)
    return enforce_dir(dir_name)


class __DummyLock:
    """A dummy replacement for locks."""

    def __enter__(self) -> '__DummyLock':
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        pass


def __log(string: str, note: str,
          stdio_lock: ContextManager) -> None:
    """
    Print a log information string to stdout.

    :param str string: the string
    :param str note: a small prefix, such as a thread ID
    :param ContextManager stdio_lock: the lock
    """
    text = f"{datetime.now()}{note}: {string}"
    with stdio_lock:
        print(text)


def __run_experiment(base_dir: str,
                     experiments: List[Tuple[Callable, Callable]],
                     n_runs: Tuple[int, ...],
                     perform_warmup: bool,
                     file_lock: ContextManager,
                     stdio_lock: ContextManager,
                     cache: Callable,
                     thread_id: str) -> None:
    """
    Execute a single thread of expeirments.

    :param str base_dir: the base directory
    :param List[Tuple[Callable, Callable]] experiments: the stream of
        experiment setups
    :param bool perform_warmup: should we perform a warm-up per instance?
    :param ContextManager file_lock: the lock for file operations
    :param ContextManager stdio_lock: the lock for log output
    :param Callable cache: the cache
    :param thread_id: the thread id
    """
    random = default_rng()

    for runs in n_runs:
        random.shuffle(experiments)

        for setup in experiments:
            instance = setup[0]()
            if instance is None:
                raise ValueError("None is not an instance.")
            inst_name = sanitize_name(str(instance))

            exp = setup[1](instance)
            if not isinstance(exp, Execution):
                raise ValueError(
                    "Setup callable must produce instance of "
                    f"Execution, but generates {type(exp)}.")
            algo_name = sanitize_name(exp.get_algorithm().get_name())

            cd = __ensure_dir(os.path.join(base_dir, algo_name, inst_name))

            seeds = rand_seeds_from_str(string=inst_name, n_seeds=runs)
            random.shuffle(seeds)
            needs_warmup = perform_warmup
            for seed in seeds:
                filename = sanitize_names([algo_name, inst_name, hex(seed)])
                log_file = os.path.join(cd, filename + FILE_SUFFIX)

                skip = True
                with file_lock:
                    if cache(log_file):
                        log_file, skip = file_ensure_exists(log_file)
                if skip:
                    continue

                exp.set_rand_seed(seed)

                if needs_warmup:
                    needs_warmup = False
                    cpy: Execution = copy.copy(exp)
                    cpy.set_max_fes(10, True)
                    cpy.set_max_time_millis(3600000, True)
                    cpy.set_log_file(None)
                    cpy.set_log_improvements(False)
                    cpy.set_log_all_fes(False)
                    __log(f"warmup for '{filename}'.", thread_id, stdio_lock)
                    with cpy.execute():
                        pass
                    del cpy

                exp.set_log_file(log_file)
                __log(filename, thread_id, stdio_lock)
                with exp.execute():
                    pass


#: The default number of threads to be used
__DEFAULT_N_THREADS: Final[int] = max(1, min(len(sched_getaffinity(0)) - 1,
                                             128))


def run_experiment(base_dir: str,
                   instances: Iterable[Callable],
                   setups: Iterable[Callable],
                   n_runs: Union[int, Iterable[int]] = 11,
                   n_threads: int = __DEFAULT_N_THREADS,
                   perform_warmup: bool = True) -> str:
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
        By default, we will use the number of processors - 1 threads
    :param bool perform_warmup: should we perform a warm-up per instance? If
        this parameter is `True`, then before the very first run of a thread on
        an instance, we will execute the algorithm for just ten function
        evaluations without logging and discard the results. The idea is that
        during this warm-up, things such as JIT compilation or complicated
        parsing can take place. While this cannot mitigate time measurement
        problems for JIT compilations taking place late in runs, it can at
        least somewhat solve the problem of delayed first FEs caused by
        compilation and parsing.

    :return: the canonicalized path to `base_dir`
    :rtype: str
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

    experiments = [(ii, ss) for ii in instances for ss in setups]

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
            raise ValueError("n_runs sequence must not be increasing and "
                             f"positive, we cannot have {run} follow {last}.")
        last = run

    cache = is_new()
    base_dir = __ensure_dir(base_dir)

    stdio_lock: ContextManager

    if n_threads > 1:
        file_lock: ContextManager = mp.Lock()
        stdio_lock = mp.Lock()
        __log(f"starting experiment with {n_threads} threads.",
              "", stdio_lock)

        processes = [mp.Process(target=__run_experiment,
                                args=(base_dir,
                                      experiments.copy(),
                                      n_runs,
                                      perform_warmup,
                                      file_lock,
                                      stdio_lock,
                                      cache,
                                      ":" + hex(i)[2:]))
                     for i in range(n_threads)]
        for i in range(n_threads):
            processes[i].start()
            __log(f"started processes {hex(i)[2:]}.", "", stdio_lock)
        for i in range(n_threads):
            processes[i].join()
            __log(f"processes {hex(i)[2:]} terminated.", "", stdio_lock)

    else:
        stdio_lock = __DummyLock()
        __run_experiment(base_dir=base_dir,
                         experiments=experiments,
                         n_runs=n_runs,
                         perform_warmup=perform_warmup,
                         file_lock=__DummyLock(),
                         stdio_lock=stdio_lock,
                         cache=cache,
                         thread_id="")

    __log("finished experiment.", "", stdio_lock)
    return base_dir
