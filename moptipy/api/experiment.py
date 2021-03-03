import os.path
from os import makedirs
import multiprocessing as mp
from typing import Iterable, Union, Callable, Tuple, List
from typing import Optional
from datetime import datetime

from numpy.random import default_rng

from math import isfinite
from moptipy.api._process_base import _check_max_fes, _check_max_time_millis, \
    _check_goal_f
from moptipy.api._process_no_ss import _ProcessNoSS
from moptipy.api._process_no_ss_log import _ProcessNoSSLog
from moptipy.api._process_ss import _ProcessSS
from moptipy.api._process_ss_log import _ProcessSSLog
from moptipy.api.algorithm import Algorithm, _check_algorithm
from moptipy.api.encoding import Encoding, _check_encoding
from moptipy.api.objective import Objective, _check_objective
from moptipy.api.process import Process
from moptipy.api.space import Space, _check_space
from moptipy.utils.cache import is_new
from moptipy.utils.io import canonicalize_path, enforce_dir
from moptipy.utils.io import file_create_or_truncate, file_ensure_exists
from moptipy.utils.logging import sanitize_name, sanitize_names
from moptipy.utils.nputils import rand_seed_check, rand_seeds_from_str


def _check_log_file(log_file: Optional[str],
                    none_is_ok: bool = True) -> Optional[str]:
    """
    Check a log file.
    :param Optional[str] log_file: the log file
    :param bool none_is_ok: is `None` ok for log files?
    :return: the log file
    :rtype: Optional[str]
    """
    if log_file is None:
        if none_is_ok:
            return None
    return canonicalize_path(log_file)


class Experiment:
    """
    This class allows us to define all the components of an experiment and
    then to execute it.
    """

    def __init__(self):
        super().__init__()
        self.__algorithm = None
        self.__solution_space = None
        self.__objective = None
        self.__search_space = None
        self.__encoding = None
        self.__rand_seed = None
        self.__max_fes = None
        self.__max_time_millis = None
        self.__goal_f = None
        self.__log_file = None
        self.__log_improvements = False
        self.__log_all_fes = False

    def set_algorithm(self, algorithm: Algorithm):
        """
        Set the algorithm to be used for this experiment.
        :param Algorithm algorithm: the algorithm
        """
        self.__algorithm = _check_algorithm(algorithm)

    def get_algorithm(self) -> Algorithm:
        """
        Obtain the algorithm. Requires that :meth:`set_algorithm` was
        called first.
        :return: the algorithm
        :rtype: Algorithm
        """
        return _check_algorithm(self.__algorithm)

    def set_solution_space(self, solution_space: Space):
        """
        Set the solution space to be used for this experiment, i.e., the space
        managing the data structure holding the candidate solutions.
        :param Space solution_space: the solution space
        """
        self.__solution_space = _check_space(solution_space)

    def set_objective(self, objective: Objective):
        """
        Set the objective function to be used for this experiment, i.e., the
        function rating the quality of candidate solutions.
        :param Objective objective: the objective function
        """
        self.__objective = _check_objective(objective)

    def set_search_space(self, search_space: Optional[Space]):
        """
        Set the search space to be used for this experiment, i.e., the
        space from which the algorithm samples points.
        :param Optional[Space] search_space: the search space, or `None` of
        none shall be used
        """
        self.__search_space = _check_space(search_space, none_is_ok=True)

    def set_encoding(self, encoding: Optional[Encoding]):
        """
        Set the encoding to be used for this experiment, i.e., the function
        translating from the search_space to the solution_space.
        :param Optional[Encoding] encoding: the encoding, or `None` of none
        shall be used
        """
        self.__encoding = _check_encoding(encoding, none_is_ok=True)

    def set_rand_seed(self, rand_seed: Optional[int]):
        """
        Set the random seed to be used for initializing the random number
        generator in the experiment.
        :param Optional[int] rand_seed: the random seed, or `None` if a seed
        should automatically be chosen when the experiment is executed
        """
        self.__rand_seed = None if rand_seed is None \
            else rand_seed_check(rand_seed)

    def set_max_fes(self, max_fes: int):
        """
        Set the maximum FEs, i.e., the number of candidate solutions an
        optimization is allowed to evaluate. If this method is called
        multiple times, then the shortest limit is used-
        :param int max_fes: the maximum FEs
        """
        max_fes = _check_max_fes(max_fes)
        if not (self.__max_fes is None):
            if max_fes >= self.__max_fes:
                return
        self.__max_fes = max_fes

    def set_max_time_millis(self, max_time_millis: int):
        """
        Set the maximum time in milliseconds that the process is allowed to
        run. If this method is called multiple times, the shortest time is
        used.
        :param int max_time_millis: the maximum time in milliseconds
        """
        max_time_millis = _check_max_time_millis(max_time_millis)
        if not (self.__max_time_millis is None):
            if max_time_millis >= self.__max_time_millis:
                return
        self.__max_time_millis = max_time_millis

    def set_goal_f(self, goal_f: Union[int, float]):
        """
        Set the goal objective value after which the process can stop. If this
        method is called multiple times, then the largest value is retained.
        :param Union[int, float] goal_f: the goal objective value.
        """
        goal_f = _check_goal_f(goal_f)
        if not (self.__goal_f is None):
            if goal_f <= self.__goal_f:
                return
        self.__goal_f = goal_f

    def set_log_file(self, log_file: str):
        """
        Set the log file to write to. This method can be called arbitrarily
        often.
        :param str log_file: the log file
        """
        self.__log_file = _check_log_file(log_file, True)

    def set_log_improvements(self, log_improvements: bool = True):
        """
        Set whether improvements should be logged
        :param bool log_improvements: if improvements should be logged?
        """
        if not isinstance(log_improvements, bool):
            raise ValueError("log improvements must be bool, but is "
                             + str(type(log_improvements)) + ".")
        self.__log_improvements = log_improvements

    def set_log_all_fes(self, log_all_fes: bool = True):
        """
        Set whether all FEs should be logged
        :param bool log_all_fes: if all FEs should be logged?
        """
        if not isinstance(log_all_fes, bool):
            raise ValueError("log all FEs  must be bool, but is "
                             + str(type(log_all_fes)) + ".")
        self.__log_all_fes = log_all_fes

    def execute(self) -> Process:
        """
        Execute the experiment and return the process.
        :return: the process that can be queried for the result
        :rtype: Process
        """
        algorithm = _check_algorithm(self.__algorithm)
        solution_space = _check_space(self.__solution_space)
        objective = _check_objective(self.__objective)
        search_space = _check_space(self.__search_space,
                                    self.__encoding is None)
        encoding = _check_encoding(self.__encoding,
                                   search_space is None)
        rand_seed = self.__rand_seed
        if not (rand_seed is None):
            rand_seed = rand_seed_check(rand_seed)
        max_time_millis = _check_max_time_millis(self.__max_time_millis,
                                                 True)
        max_fes = _check_max_fes(self.__max_fes, True)
        goal_f = _check_goal_f(self.__goal_f, True)
        f_lb = objective.lower_bound()
        if (not (f_lb is None)) and isfinite(f_lb):
            if (goal_f is None) or (f_lb > goal_f):
                goal_f = f_lb

        log_all_fes = self.__log_all_fes
        log_improvements = self.__log_improvements or self.__log_all_fes

        log_file = self.__log_file
        if log_file is None:
            if log_all_fes:
                raise ValueError("Log file cannot be None "
                                 "if all FEs should be logged.")
            if log_improvements:
                raise ValueError("Log file cannot be None "
                                 "if improvements should be logged.")

        else:
            log_file = file_create_or_truncate(log_file)

        if search_space is None:
            if log_improvements or log_all_fes:
                process = _ProcessNoSSLog(solution_space=solution_space,
                                          objective=objective,
                                          algorithm=algorithm,
                                          log_file=log_file,
                                          rand_seed=rand_seed,
                                          max_fes=max_fes,
                                          max_time_millis=max_time_millis,
                                          goal_f=goal_f,
                                          log_improvements=log_improvements,
                                          log_all_fes=log_all_fes)
            else:
                process = _ProcessNoSS(solution_space=solution_space,
                                       objective=objective,
                                       algorithm=algorithm,
                                       log_file=log_file,
                                       rand_seed=rand_seed,
                                       max_fes=max_fes,
                                       max_time_millis=max_time_millis,
                                       goal_f=goal_f)
        else:
            if log_improvements or log_all_fes:
                process = _ProcessSSLog(
                    solution_space=solution_space,
                    objective=objective,
                    algorithm=algorithm,
                    search_space=search_space,
                    encoding=encoding,
                    log_file=log_file,
                    rand_seed=rand_seed,
                    max_fes=max_fes,
                    max_time_millis=max_time_millis,
                    goal_f=goal_f,
                    log_improvements=log_improvements,
                    log_all_fes=log_all_fes)
            else:
                process = _ProcessSS(solution_space=solution_space,
                                     objective=objective,
                                     algorithm=algorithm,
                                     search_space=search_space,
                                     encoding=encoding,
                                     log_file=log_file,
                                     rand_seed=rand_seed,
                                     max_fes=max_fes,
                                     max_time_millis=max_time_millis,
                                     goal_f=goal_f)
        # noinspection PyProtectedMember
        process._after_init()
        algorithm.solve(process)
        return process


def __ensure_dir(dir_name: str) -> str:
    dir_name = canonicalize_path(dir_name)
    makedirs(name=dir_name, exist_ok=True)
    return enforce_dir(dir_name)


class __DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass


def __log(string: str, note: str, stdio_lock: Union[mp.Lock, __DummyLock]):
    text = str(datetime.now()) + note + ": " + string
    with stdio_lock:
        print(text)


def __run_experiment(base_dir: str,
                     experiments: List[Tuple],
                     n_runs: Tuple[int],
                     file_lock: Union[mp.Lock, __DummyLock],
                     stdio_lock: Union[mp.Lock, __DummyLock],
                     cache,
                     note: str) -> None:
    random = default_rng()

    for runs in n_runs:
        random.shuffle(experiments)

        for setup in experiments:
            instance = setup[0]()
            if instance is None:
                raise ValueError("None is not an instance.")
            inst_name = sanitize_name(str(instance))

            exp = setup[1](instance)
            if not isinstance(exp, Experiment):
                raise ValueError(
                    "Setup callable must produce instance of "
                    "Experiment, but generates " + str(type(exp)) + ".")
            algo_name = sanitize_name(exp.get_algorithm().get_name())

            cd = __ensure_dir(os.path.join(base_dir, algo_name, inst_name))

            seeds = rand_seeds_from_str(string=inst_name, n_seeds=runs)
            random.shuffle(seeds)
            for seed in seeds:
                filename = sanitize_names([algo_name, inst_name, hex(seed)])
                log_file = os.path.join(cd, filename + ".txt")

                skip = True
                with file_lock:
                    if cache(log_file):
                        log_file, skip = file_ensure_exists(log_file)
                if skip:
                    continue

                __log(filename, note, stdio_lock)

                exp.set_rand_seed(seed)
                exp.set_log_file(log_file)
                with exp.execute():
                    pass


def run_experiment(base_dir: str,
                   instances: Iterable[Callable],
                   setups: Iterable[Callable],
                   n_runs: Union[int, Iterable[int]] = 11,
                   n_threads: int = 1) -> str:
    """
    Run an experiment and store the log files into the given folder.
    :param base_dir: the base directory where to store the results
    :param instances: an iterable of callables, each of which should return
    an object representing a problem instance, whose `str(..)` representation
    is a valid name
    :param setups: an iterable of callables, each receiving an instance (as
    returned by instances) as input and producing an :class:`Experiment` as
    output
    :param n_runs: the number of runs per algorithm - instance combination
    :param n_threads: the number of parallel threads of execution to use
    :return: the canonicalized path to `base_dir`
    :rtype: str
    """
    if not isinstance(instances, Iterable):
        raise TypeError("instances must be a iterable object, but is "
                        + str(type(instances)) + ".")
    if not isinstance(setups, Iterable):
        raise TypeError("setups must be a iterable object, but is "
                        + str(type(setups)) + ".")
    if not isinstance(n_threads, int):
        raise TypeError("n_threads must be int, but is "
                        + str(type(n_threads)) + ".")
    if n_threads <= 0:
        raise ValueError("n_threads must be positive, but is "
                         + str(n_threads) + ".")

    instances = list(instances)
    if len(instances) <= 0:
        raise ValueError("Instance enumeration is empty.")
    for instance in instances:
        if not isinstance(instance, Callable):
            raise ValueError(
                "All instances must be Callables, but encountered a "
                + str(type(instance)) + ".")

    setups = list(setups)
    if len(setups) <= 0:
        raise ValueError("Setup enumeration is empty.")
    for setup in setups:
        if not isinstance(setup, Callable):
            raise ValueError("All setups must be Callables, but encountered a "
                             + str(type(setup)) + ".")

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
    for r in n_runs:
        if r is None:
            raise ValueError("n_runs must not be None.")
        if r <= last:
            raise ValueError("n_runs sequence must not be increasing and "
                             "positive, we cannot have " + str(r)
                             + " follow " + str(last) + ".")
        last = r

    cache = is_new()
    base_dir = __ensure_dir(base_dir)

    if n_threads > 1:
        file_lock = mp.Lock()
        stdio_lock = mp.Lock()
        __log("starting experiment with " + str(n_threads) + " threads.",
              "", stdio_lock)

        processes = [mp.Process(target=__run_experiment,
                                args=(base_dir,
                                      experiments.copy(),
                                      n_runs,
                                      file_lock,
                                      stdio_lock,
                                      cache,
                                      ":" + hex(i)[2:]))
                     for i in range(n_threads)]
        for i in range(n_threads):
            processes[i].start()
            __log("started processes " + hex(i)[2:] + ".", "", stdio_lock)
        for i in range(n_threads):
            processes[i].join()
            __log("processes " + hex(i)[2:] + " terminated.", "", stdio_lock)

    else:
        stdio_lock = __DummyLock()
        __run_experiment(base_dir=base_dir,
                         experiments=experiments,
                         n_runs=n_runs,
                         file_lock=__DummyLock(),
                         stdio_lock=stdio_lock,
                         cache=cache,
                         note="")

    __log("finished experiment.", "", stdio_lock)
    return base_dir
