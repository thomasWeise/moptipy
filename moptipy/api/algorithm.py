import os
from abc import ABC, abstractmethod
from os.path import realpath, isfile
from typing import Union, Optional, Callable

from moptipy.api.process import Process
from moptipy.api.component import Component
from moptipy.api.objective import Objective, _CallableComponent
from moptipy.api.encoding import Encoding
from moptipy.api.space import Space
from moptipy.api._process_no_ss import _ProcessNoSS
from moptipy.api._process_ss import _ProcessSS
from moptipy.api._process_no_ss_log import _ProcessNoSSLog
from moptipy.api._process_ss_log import _ProcessSSLog
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.utils.logger import KeyValueSection
from moptipy.utils import logging


class Algorithm(Component):
    """A class to represent an optimization algorithm."""

    @abstractmethod
    def solve(self, process: Process) -> None:
        """
        Apply this optimization algorithm to the given black-box process

        :param moptipy.api.Process process: the black-box process
        """
        raise NotImplementedError


class Algorithm0(Algorithm, ABC):
    """An algorithm implementation with a nullary search operator."""

    def __init__(self,
                 op0: Op0,
                 op0_is_default: bool = True):
        """
        Create the algorithm with nullary search operator
        :param moptipy.api.Op0 op0: the nullary search operator
        :param bool op0_is_default: is this a default nullary operator?
            if yes, it will not be included in the name
        """
        if (op0 is None) or (not isinstance(op0, Op0)):
            ValueError("op0 must be instance of Op0, but is '"
                       + str(type(op0)) + "'")
        self.op0 = op0
        self._op0_is_default = op0_is_default

    def get_name(self):
        return "" if self._op0_is_default else self.op0.get_name()

    def log_parameters_to(self, logger: KeyValueSection):
        super().log_parameters_to(logger)
        with logger.scope(logging.SCOPE_OP0) as sc:
            self.op0.log_parameters_to(sc)


class Algorithm1(Algorithm0, ABC):
    """An algorithm implementation with a unary search operator."""

    def __init__(self,
                 op0: Op0,
                 op1: Op1,
                 op0_is_default: bool = True,
                 op1_is_default: bool = False):
        """
        Create the algorithm with nullary and unary search operator
        :param moptipy.api.Op0 op0: the nullary search operator
        :param moptipy.api.Op1 op1: the unary search operator
        :param bool op0_is_default: is this a default nullary operator?
            if yes, it will not be included in the name
        :param bool op1_is_default: is this a default unary operator?
            if yes, it will not be included in the name
        """
        super().__init__(op0=op0,
                         op0_is_default=op0_is_default)
        if (op1 is None) or (not isinstance(op1, Op1)):
            ValueError("op1 must be instance of Op1, but is '"
                       + str(type(op1)) + "'")
        self.op1 = op1
        self._op1_is_default = op1_is_default

    def get_name(self):
        return logging.sanitize_names([
            "" if self._op0_is_default else self.op0.get_name(),
            "" if self._op1_is_default else self.op1.get_name()])

    def log_parameters_to(self, logger: KeyValueSection):
        super().log_parameters_to(logger)
        with logger.scope(logging.SCOPE_OP1) as sc:
            self.op1.log_parameters_to(sc)


class Algorithm2(Algorithm1, ABC):
    """
    An algorithm implementation with a binary and unary search operator.
    """

    def __init__(self,
                 op0: Op0,
                 op1: Op1,
                 op2: Op2,
                 op0_is_default: bool = True,
                 op1_is_default: bool = False,
                 op2_is_default: bool = False):
        """
        Create the algorithm with nullary and unary search operator
        :param moptipy.api.Op0 op0: the nullary search operator
        :param moptipy.api.Op1 op1: the unary search operator
        :param moptipy.api.Op2 op2: the binary search operator
        :param bool op0_is_default: is this a default nullary operator?
            if yes, it will not be included in the name
        :param bool op1_is_default: is this a default unary operator?
            if yes, it will not be included in the name
        :param bool op2_is_default: is this a default binary operator?
            if yes, it will not be included in the name
        """
        super().__init__(op0=op0,
                         op1=op1,
                         op0_is_default=op0_is_default,
                         op1_is_default=op1_is_default)
        if (op2 is None) or (not isinstance(op2, Op2)):
            ValueError("op2 must be instance of Op2, but is '"
                       + str(type(op2)) + "'")
        self.op2 = op2
        self._op2_is_default = op2_is_default

    def get_name(self):
        return logging.sanitize_names([
            "" if self._op0_is_default else self.op0.get_name(),
            "" if self._op1_is_default else self.op1.get_name(),
            "" if self._op2_is_default else self.op2.get_name()])

    def log_parameters_to(self, logger: KeyValueSection):
        super().log_parameters_to(logger)
        with logger.scope(logging.SCOPE_OP2) as sc:
            self.op2.log_parameters_to(sc)


class CallableAlgorithm(_CallableComponent, Algorithm):
    """
    Wrapping a Callable such as a lambda into an algorithm.
    """

    def __init__(self,
                 algorithm: Callable,
                 name: str = None):
        """
        Create a wrapper mapping a Callable to an objective function

        :param Callable algorithm: the function to wrap, can be a
            lambda expression
        :param str name: the name of the algorithm
        """
        super().__init__(inner=algorithm,
                         name="unnamed_algorithm" if (name is None) else name)

    def solve(self, process: Process) -> None:
        self._inner(process)


def solve(algorithm: Algorithm,
          solution_space: Space,
          objective_function: Objective,
          search_space: Space = None,
          encoding: Encoding = None,
          rand_seed: Optional[int] = None,
          max_time_millis: Optional[int] = None,
          max_fes: Optional[int] = None,
          goal_f: Union[int, float, None] = None,
          log_file: Optional[str] = None,
          log_improvements: bool = False,
          log_all_fes: bool = False,
          log_state: bool = False,
          overwrite_log: bool = False) -> Process:
    """
    Apply an optimization algorithm to the given problem description.
    This means that an instance :class:`~moptipy.api.Process` will be
    constructed. Then the method :func:`~moptipy.api.Algorithm.solve`
    will be invoked. It may optionally log the progress and results to
    a file ``log_file``. Finally, the :class:`~moptipy.api.Process`
    instance will be returned.
    :param Algorithm algorithm: the algorithm to apply
    :param solution_space: the solution space
    :param objective_function: the objective function
    :param search_space: the optional search space
    :param encoding: the representation mapping
    :param rand_seed: the random seed
    :param max_time_millis: the maximum runtime permitted, in
    milliseconds
    :param max_fes: the maximum objective function evaluations
    permitted
    :param goal_f: the goal objective value
    :param log_file: the optional log file
    :param bool log_improvements: should we log improvements?
    :param bool log_all_fes: should we log all fes
    :param bool log_state: should we be able to log dynamic state?
    :param bool overwrite_log: can the log file be overwritten
    if it exists?
    :return: the :class:`~moptipy.api.Process` instance which can be
    queried for the final result
    """

    if not (isinstance(algorithm, Algorithm)):
        raise ValueError("Optimization algorithm needs to be specified, "
                         "but found '"
                         + str(type(algorithm)) + "'.")

    if search_space is None:
        if not (encoding is None):
            raise ValueError("If search_space is not provided, neither can "
                             "encoding.")
    elif encoding is None:
        if search_space == solution_space:
            search_space = None
        else:
            raise ValueError("If a search_space != solution_space is "
                             "provided, encoding must be "
                             "provided, too.")

    if log_file is None:
        if log_all_fes or log_improvements or log_state:
            raise ValueError("Can only log stuff if log file is specified")
    else:
        log_file = realpath(log_file)
        try:
            os.close(os.open(log_file,
                             os.O_CREAT if overwrite_log else
                             (os.O_CREAT | os.O_EXCL)))
        except FileExistsError as err:
            raise ValueError("Log file '" + log_file
                             + "' already exists.") from err
        except Exception as err:
            raise ValueError("Error when creating log file'"
                             + log_file + "'.") from err
        if not isfile(log_file):
            raise ValueError("Creation of file '"
                             + log_file + "' did not produce file?")

    if search_space is None:
        if log_improvements or log_all_fes or log_state:
            process = _ProcessNoSSLog(solution_space=solution_space,
                                      objective_function=objective_function,
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
                                   objective_function=objective_function,
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
                objective_function=objective_function,
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
                                 objective_function=objective_function,
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
