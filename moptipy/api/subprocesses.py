"""Different ways to transform and slice processes."""
from typing import Final, Callable, Union, Any, TypeVar, Optional

import numpy as np

from moptipy.api.mo_process import MOProcess
from moptipy.api.process import Process, check_max_fes
from moptipy.utils.types import type_error

#: the type variable for single- and multi-objective processes.
T = TypeVar('T', Process, MOProcess)


class __ForFEs(Process):
    """A process searching for a fixed amount of FEs."""

    def __init__(self, owner: Process, max_fes: int):
        super().__init__()
        if not isinstance(owner, Process):
            raise type_error(owner, "owner", Process)
        #: the owning process
        self._owner: Final[Process] = owner
        self.get_random = owner.get_random  # type: ignore
        a = owner.get_consumed_time_millis  # type: ignore
        self.get_consumed_time_millis = a  # type: ignore
        a = owner.get_max_time_millis  # type: ignore
        self.get_max_time_millis = a  # type: ignore
        a = owner.get_last_improvement_time_millis  # type: ignore
        self.get_last_improvement_time_millis = a  # type: ignore
        self.add_log_section = owner.add_log_section  # type: ignore
        self.lower_bound = owner.lower_bound  # type: ignore
        self.upper_bound = owner.upper_bound  # type: ignore
        self.create = owner.create  # type: ignore
        self.copy = owner.copy  # type: ignore
        self.to_str = owner.to_str  # type: ignore
        self.is_equal = owner.is_equal  # type: ignore
        self.from_str = owner.from_str  # type: ignore
        self.validate = owner.validate  # type: ignore
        self.n_points = owner.n_points  # type: ignore
        self.has_best = owner.has_best  # type: ignore
        self.get_copy_of_best_x = owner.get_copy_of_best_x  # type: ignore
        self.get_best_f = owner.get_best_f  # type: ignore
        #: the maximum FEs
        self.max_fes: Final[int] = check_max_fes(max_fes)
        #: the FEs that we still have left
        self.__fes_left: int = max_fes
        #: did we terminate?
        self.__terminated: bool = False
        #: the fast call to the owner's should_terminate method
        self.__should_terminate: Final[Callable[[], bool]] \
            = owner.should_terminate
        #: the fast call to the owner's evaluate method
        self.__evaluate: Final[Callable[[Any], Union[int, float]]] \
            = owner.evaluate
        #: the fast call to the owner's register method
        self.__register: Final[Callable[[Any, Union[int, float]], None]] \
            = owner.register
        #: the start fe
        self.__start_fe: Final[int] = owner.get_consumed_fes()

    def should_terminate(self) -> bool:
        return self.__terminated or self.__should_terminate()

    def terminate(self) -> None:
        self.__terminated = True

    def evaluate(self, x) -> Union[float, int]:
        f: Final[Union[int, float]] = self.__evaluate(x)
        fel: Final[int] = self.__fes_left - 1
        self.__fes_left = fel
        if fel <= 0:
            self.__terminated = True
        return f

    def register(self, x, f: Union[int, float]) -> None:
        self.__register(x, f)
        fel: Final[int] = self.__fes_left - 1
        self.__fes_left = fel
        if fel <= 0:
            self.__terminated = True

    def get_consumed_fes(self) -> int:
        return self.max_fes - self.__fes_left

    def get_last_improvement_fe(self) -> int:
        return max(1 if self.__fes_left < self.max_fes else 0,
                   self._owner.get_last_improvement_fe() - self.__start_fe)

    def get_max_fes(self) -> int:
        return self.max_fes

    def __str__(self) -> str:
        return f"forFEs_{self.max_fes}_{self._owner}"


class __ForFEsMO(MOProcess):
    """A process searching for a fixed amount of FEs."""

    def __init__(self, owner: MOProcess, max_fes: int):
        super().__init__()
        if not isinstance(owner, MOProcess):
            raise type_error(owner, "owner", MOProcess)
        #: the owning process
        self._owner: Final[MOProcess] = owner
        self.get_random = owner.get_random  # type: ignore
        a = owner.get_consumed_time_millis  # type: ignore
        self.get_consumed_time_millis = a  # type: ignore
        a = owner.get_max_time_millis  # type: ignore
        self.get_max_time_millis = a  # type: ignore
        a = owner.get_last_improvement_time_millis  # type: ignore
        self.get_last_improvement_time_millis = a  # type: ignore
        self.add_log_section = owner.add_log_section  # type: ignore
        self.lower_bound = owner.lower_bound  # type: ignore
        self.upper_bound = owner.upper_bound  # type: ignore
        self.create = owner.create  # type: ignore
        self.copy = owner.copy  # type: ignore
        self.to_str = owner.to_str  # type: ignore
        self.is_equal = owner.is_equal  # type: ignore
        self.from_str = owner.from_str  # type: ignore
        self.validate = owner.validate  # type: ignore
        self.n_points = owner.n_points  # type: ignore
        self.has_best = owner.has_best  # type: ignore
        self.get_copy_of_best_x = owner.get_copy_of_best_x  # type: ignore
        self.get_best_f = owner.get_best_f  # type: ignore
        self.get_archive = owner.get_archive  # type: ignore
        self.check_in = owner.check_in  # type: ignore
        self.f_create = owner.f_create  # type: ignore
        self.f_validate = owner.f_validate  # type: ignore
        self.f_dtype = owner.f_dtype  # type: ignore
        self.f_dominates = owner.f_dominates  # type: ignore
        self.f_dimension = owner.f_dimension  # type: ignore
        #: the maximum FEs
        self.max_fes: Final[int] = check_max_fes(max_fes)
        #: the FEs that we still have left
        self.__fes_left: int = max_fes
        #: did we terminate?
        self.__terminated: bool = False
        #: the fast call to the owner's should_terminate method
        self.__should_terminate: Final[Callable[[], bool]] \
            = owner.should_terminate
        #: the fast call to the owner's evaluate method
        self.__evaluate: Final[Callable[[Any], Union[int, float]]] \
            = owner.evaluate
        #: the fast call to the owner's register method
        self.__register: Final[Callable[[Any, Union[int, float]], None]] \
            = owner.register
        #: the evaluation wrapper
        self.__f_evaluate: Final[Callable[
            [Any, np.ndarray], Union[int, float]]] = owner.f_evaluate
        #: the start fe
        self.__start_fe: Final[int] = owner.get_consumed_fes()

    def should_terminate(self) -> bool:
        return self.__terminated or self.__should_terminate()

    def terminate(self) -> None:
        self.__terminated = True

    def evaluate(self, x) -> Union[float, int]:
        f: Final[Union[int, float]] = self.__evaluate(x)
        fel: Final[int] = self.__fes_left - 1
        self.__fes_left = fel
        if fel <= 0:
            self.__terminated = True
        return f

    def register(self, x, f: Union[int, float]) -> None:
        self.__register(x, f)
        fel: Final[int] = self.__fes_left - 1
        self.__fes_left = fel
        if fel <= 0:
            self.__terminated = True

    def f_evaluate(self, x, fs: np.ndarray) -> Union[float, int]:
        f: Final[Union[int, float]] = self.__f_evaluate(x, fs)
        fel: Final[int] = self.__fes_left - 1
        self.__fes_left = fel
        if fel <= 0:
            self.__terminated = True
        return f

    def get_consumed_fes(self) -> int:
        return self.max_fes - self.__fes_left

    def get_last_improvement_fe(self) -> int:
        return max(1 if self.__fes_left < self.max_fes else 0,
                   self._owner.get_last_improvement_fe() - self.__start_fe)

    def get_max_fes(self) -> int:
        return self.max_fes

    def __str__(self) -> str:
        return f"forFEsMO_{self.max_fes}_{self._owner}"


def for_fes(process: T, max_fes: int) -> T:
    """
    Create a sub-process that can run for the given number of FEs.

    :param process: the original process
    :param max_fes: the maximum number of objective function evaluations
    :returns: the sub-process that will terminate after `max_fes` FEs and that
        forwards all other calls the `process`.
    """
    if not isinstance(max_fes, int):
        raise type_error(max_fes, "max_fes", int)
    if max_fes <= 0:
        raise ValueError(f"max_fes cannot be {max_fes}!")
    return __ForFEsMO(process, max_fes) if isinstance(process, MOProcess) \
        else __ForFEs(process, max_fes)


class __FromStatingPoint(Process):
    """A process searching from a given point."""

    def __init__(self, owner: Process, in_and_out_x: Any,
                 f: Union[int, float]):
        """
        Create a sub-process searching from one starting point.

        :param owner: the owning process
        :param in_and_out_x: the input solution record, which will be
            overwritten with the best encountered solution
        :param f: the objective value corresponding to `in_and_out`
        """
        super().__init__()
        if not isinstance(owner, Process):
            raise type_error(owner, "owner", Process)
        #: the owning process
        self._owner: Final[Process] = owner
        self.get_random = owner.get_random  # type: ignore
        a = owner.get_consumed_time_millis  # type: ignore
        self.get_consumed_time_millis = a  # type: ignore
        a = owner.get_max_time_millis  # type: ignore
        self.get_max_time_millis = a  # type: ignore
        a = owner.get_last_improvement_time_millis  # type: ignore
        self.get_last_improvement_time_millis = a  # type: ignore
        self.add_log_section = owner.add_log_section  # type: ignore
        self.lower_bound = owner.lower_bound  # type: ignore
        self.upper_bound = owner.upper_bound  # type: ignore
        self.create = owner.create  # type: ignore
        self.copy = owner.copy  # type: ignore
        self.to_str = owner.to_str  # type: ignore
        self.is_equal = owner.is_equal  # type: ignore
        self.from_str = owner.from_str  # type: ignore
        self.validate = owner.validate  # type: ignore
        self.n_points = owner.n_points  # type: ignore
        self.should_terminate = owner.should_terminate  # type: ignore
        self.terminate = owner.terminate  # type: ignore
        #: the best solution
        self.__best_x: Final[Any] = in_and_out_x
        #: the best-so-far solution
        self.__best_f: Union[int, float] = f
        #: the last improvement fe
        self.__last_improvement_fe: int = 0
        #: the consumed FEs
        self.__fes: int = 0
        mfes: Optional[int] = owner.get_max_fes()
        if mfes is not None:
            mfes = mfes - owner.get_consumed_fes()
        #: the maximum permitted FEs
        self.__max_fes: Final[Optional[int]] = mfes
        #: the fast call to the owner's evaluate method
        self.__evaluate: Final[Callable[[Any], Union[int, float]]] \
            = owner.evaluate
        #: the fast call to the owner's register method
        self.__register: Final[Callable[[Any, Union[int, float]], None]] \
            = owner.register

    def has_best(self) -> bool:
        return True

    def get_copy_of_best_x(self, x) -> None:
        self.copy(x, self.__best_x)

    def get_best_f(self) -> Union[int, float]:
        return self.__best_f

    def evaluate(self, x) -> Union[float, int]:
        self.__fes = fe = self.__fes + 1
        f: Final[Union[int, float]] = self.__evaluate(x)
        if f < self.__best_f:
            self.__best_f = f
            self.copy(self.__best_x, x)
            self.__last_improvement_fe = fe
        return f

    def register(self, x, f: Union[int, float]) -> None:
        self.__register(x, f)
        self.__fes = fe = self.__fes + 1
        if f < self.__best_f:
            self.__best_f = f
            self.copy(self.__best_x, x)
            self.__last_improvement_fe = fe

    def get_consumed_fes(self) -> int:
        return max(1, self.__fes)

    def get_last_improvement_fe(self) -> int:
        return self.__last_improvement_fe

    def get_max_fes(self) -> Optional[int]:
        return self.__max_fes

    def __str__(self) -> str:
        return f"fromStart_{self._owner}"


def from_starting_point(owner: Process, in_and_out_x: Any,
                        f: Union[int, float]) -> Process:
    """
    Create a sub-process searching from one starting point.

    :param owner: the owning process
    :param in_and_out_x: the input solution record, which will be
        overwritten with the best encountered solution
    :param f: the objective value corresponding to `in_and_out`
    """
    return __FromStatingPoint(owner, in_and_out_x, f)


class _InternalTerminationError(Exception):
    """A protected internal termination error."""


class __WithoutShouldTerminate(Process):
    """A process allowing algorithm execution ignoring `should_terminate`."""

    def __init__(self, owner: Process):
        super().__init__()
        if not isinstance(owner, Process):
            raise type_error(owner, "owner", Process)
        #: the owning process
        self._owner: Final[Process] = owner
        self.get_random = owner.get_random  # type: ignore
        a = owner.get_consumed_time_millis  # type: ignore
        self.get_consumed_time_millis = a  # type: ignore
        a = owner.get_max_time_millis  # type: ignore
        self.get_max_time_millis = a  # type: ignore
        a = owner.get_last_improvement_time_millis  # type: ignore
        self.get_last_improvement_time_millis = a  # type: ignore
        a = owner.get_last_improvement_fe  # type: ignore
        self.get_last_improvement_fe = a  # type: ignore
        self.add_log_section = owner.add_log_section  # type: ignore
        self.lower_bound = owner.lower_bound  # type: ignore
        self.upper_bound = owner.upper_bound  # type: ignore
        self.create = owner.create  # type: ignore
        self.copy = owner.copy  # type: ignore
        self.to_str = owner.to_str  # type: ignore
        self.is_equal = owner.is_equal  # type: ignore
        self.from_str = owner.from_str  # type: ignore
        self.validate = owner.validate  # type: ignore
        self.n_points = owner.n_points  # type: ignore
        self.has_best = owner.has_best  # type: ignore
        self.get_copy_of_best_x = owner.get_copy_of_best_x  # type: ignore
        self.get_best_f = owner.get_best_f  # type: ignore
        self.should_terminate = owner.should_terminate  # type: ignore
        self.terminate = owner.terminate  # type: ignore
        self.get_max_fes = owner.get_max_fes  # type: ignore
        self.get_consumed_fes = owner.get_consumed_fes  # type: ignore
        #: the fast call to the owner's evaluate method
        self.__evaluate: Final[Callable[[Any], Union[int, float]]] \
            = owner.evaluate
        #: the fast call to the owner's register method
        self.__register: Final[Callable[[Any, Union[int, float]], None]] \
            = owner.register

    def evaluate(self, x) -> Union[float, int]:
        if self.should_terminate():
            raise _InternalTerminationError()
        return self.__evaluate(x)

    def register(self, x, f: Union[int, float]) -> None:
        if self.should_terminate():
            raise _InternalTerminationError()
        self.__register(x, f)

    def __str__(self) -> str:
        return f"protect_{self._owner}"

    def __exit__(self, exception_type, exception_value, traceback) -> bool:
        return True


class __WithoutShouldTerminateMO(MOProcess):
    """A process allowing algorithm execution ignoring `should_terminate`."""

    def __init__(self, owner: Process):
        super().__init__()
        if not isinstance(owner, MOProcess):
            raise type_error(owner, "owner", MOProcess)
        #: the owning process
        self._owner: Final[MOProcess] = owner
        self.get_random = owner.get_random  # type: ignore
        a = owner.get_consumed_time_millis  # type: ignore
        self.get_consumed_time_millis = a  # type: ignore
        a = owner.get_max_time_millis  # type: ignore
        self.get_max_time_millis = a  # type: ignore
        a = owner.get_last_improvement_time_millis  # type: ignore
        self.get_last_improvement_time_millis = a  # type: ignore
        a = owner.get_last_improvement_fe  # type: ignore
        self.get_last_improvement_fe = a  # type: ignore
        self.add_log_section = owner.add_log_section  # type: ignore
        self.lower_bound = owner.lower_bound  # type: ignore
        self.upper_bound = owner.upper_bound  # type: ignore
        self.create = owner.create  # type: ignore
        self.copy = owner.copy  # type: ignore
        self.to_str = owner.to_str  # type: ignore
        self.is_equal = owner.is_equal  # type: ignore
        self.from_str = owner.from_str  # type: ignore
        self.validate = owner.validate  # type: ignore
        self.n_points = owner.n_points  # type: ignore
        self.has_best = owner.has_best  # type: ignore
        self.get_copy_of_best_x = owner.get_copy_of_best_x  # type: ignore
        self.get_best_f = owner.get_best_f  # type: ignore
        self.should_terminate = owner.should_terminate  # type: ignore
        self.terminate = owner.terminate  # type: ignore
        self.get_max_fes = owner.get_max_fes  # type: ignore
        self.get_consumed_fes = owner.get_consumed_fes  # type: ignore
        self.f_create = owner.f_create  # type: ignore
        self.f_validate = owner.f_validate  # type: ignore
        self.f_dtype = owner.f_dtype  # type: ignore
        self.f_dominates = owner.f_dominates  # type: ignore
        self.f_dimension = owner.f_dimension  # type: ignore
        self.get_archive = owner.get_archive  # type: ignore
        self.check_in = owner.check_in  # type: ignore
        #: the fast call to the owner's evaluate method
        self.__evaluate: Final[Callable[[Any], Union[int, float]]] \
            = owner.evaluate
        #: the fast call to the owner's register method
        self.__register: Final[Callable[[Any, Union[int, float]], None]] \
            = owner.register
        #: the fast call to the owner's f_evaluate method
        self.__f_evaluate: Final[Callable[
            [Any, np.ndarray], Union[int, float]]] \
            = owner.f_evaluate

    def evaluate(self, x) -> Union[float, int]:
        if self.should_terminate():
            raise _InternalTerminationError()
        return self.__evaluate(x)

    def register(self, x, f: Union[int, float]) -> None:
        if self.should_terminate():
            raise _InternalTerminationError()
        self.__register(x, f)

    def f_evaluate(self, x, fs: np.ndarray) -> Union[int, float]:
        if self.should_terminate():
            raise _InternalTerminationError()
        return self.__f_evaluate(x, fs)

    def __str__(self) -> str:
        return f"protectMO_{self._owner}"

    def __exit__(self, exception_type, exception_value, traceback) -> bool:
        return True


def without_should_terminate(algorithm: Callable[[T], Any], process: T):
    """
    Apply an algorithm that does not call `should_terminate` to a process.

    If we use an algorithm from an external library, this algorithm may ignore
    the proper usage of our API. With this method, we try to find a way to
    make sure that these calls are consistent with the termination criterion
    of the `moptipy` API.

    Before calling :meth:`~moptipy.api.process.Process.evaluate`,
    :meth:`~moptipy.api.process.Process.register`, or
    :meth:`~moptipy.api.mo_problem.MOProblem.f_evaluate`, an optimization
    algorithm must check if it should instead stop via
    :meth:`~moptipy.api.process.Process.should_terminate`. If the process
    called :meth:`~moptipy.api.process.Process.should_terminate` and was
    told to stop but did invoke the evaluation routines anyway, an
    exception will be thrown and the process force-terminates. If the
    process did not call
    :meth:`~moptipy.api.process.Process.should_terminate` but was
    supposed to stop, the results of
    :meth:`~moptipy.api.process.Process.evaluate` may be arbitrary (or
    positive infinity).

    This function here can be used to deal with processes that do not invoke
    :meth:`~moptipy.api.process.Process.should_terminate`. It will invoke
    this method by itself before
    :meth:`~moptipy.api.process.Process.evaluate`,
    :meth:`~moptipy.api.process.Process.register`, and
    :meth:`~moptipy.api.mo_problem.MOProblem.f_evaluate` and terminate the
    algorithm with an exception if necessary. It will then catch the
    exception and bury it.

    Thus, we can now use algorithms that ignore our termination criteria and
    still force them to terminate when they should.

    :param algorithm: the algorithm
    :param process: the optimization process
    """
    try:
        with __WithoutShouldTerminateMO(process) \
                if isinstance(process, MOProcess) \
                else __WithoutShouldTerminate(process) as proc:
            algorithm(proc)
    except _InternalTerminationError:
        pass  # the internal error is ignored
