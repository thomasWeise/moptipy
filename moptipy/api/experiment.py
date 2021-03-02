from typing import Optional, Union

from moptipy.api._process_base import _check_max_fes, _check_max_time_millis,\
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
from moptipy.utils.io import canonicalize_path
from moptipy.utils.io import file_create_or_truncate
from moptipy.utils.nputils import rand_seed_check


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
        self.__log_state = False

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

    def set_log_state(self, log_state: bool = True):
        """
        Set whether dynamic state should be logged
        :param bool log_state: if dynamic state should be logged
        """
        if not isinstance(log_state, bool):
            raise ValueError("log state  must be bool, but is "
                             + str(type(log_state)) + ".")
        self.__log_state = log_state

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

        log_all_fes = self.__log_all_fes
        log_improvements = self.__log_improvements or self.__log_all_fes
        log_state = self.__log_state

        log_file = self.__log_file
        if log_file is None:
            if log_all_fes:
                raise ValueError("Log file cannot be None "
                                 "if all FEs should be logged.")
            if log_improvements:
                raise ValueError("Log file cannot be None "
                                 "if improvements should be logged.")
            if log_state:
                raise ValueError("Log file cannot be None "
                                 "if dynamic state should be logged.")

        else:
            log_file = file_create_or_truncate(log_file)

        if search_space is None:
            if log_improvements or log_all_fes or log_state:
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
            if log_improvements or log_all_fes or log_state:
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
