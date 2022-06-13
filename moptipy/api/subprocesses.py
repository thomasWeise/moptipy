"""Different ways to transform and slice processes."""
from typing import Final, Callable, Union, Any

from moptipy.api.process import Process, check_max_fes
from moptipy.utils.types import type_error


class ForFEs(Process):
    """A process searching for a fixed amount of FEs."""

    def __init__(self, owner: Process, max_fes: int):
        """
        Create a sub-process searching for some FEs.

        :param owner: the owning process`
        :param max_fes: the maximum number of FEs granted to the sub-process
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
        self.has_best = owner.has_best  # type: ignore
        self.get_copy_of_best_x = owner.get_copy_of_best_x  # type: ignore
        self.get_best_f = owner.get_best_f  # type: ignore
        #: the maximum FEs
        self.max_fes: Final[int] = check_max_fes(max_fes)
        #: the FEs that we still have left
        self.__fes_left: int = max_fes
        #: did we terminate?
        self.__terminated: bool = False
        #: the fast call to the owner's shouldTerminate method
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
        """
        Check if this sub-process should terminate.

        :returns: `True` if the allotted FEs have been consumed
        """
        return self.__terminated or self.__should_terminate()

    def terminate(self) -> None:
        """Terminate this sub-process."""
        self.__terminated = True

    def evaluate(self, x) -> Union[float, int]:
        """
        Evaluate a solution and return the objective value.

        :param x: the solution
        :returns: the corresponding objective value
        """
        f: Final[Union[int, float]] = self.__evaluate(x)
        fel: Final[int] = self.__fes_left - 1
        self.__fes_left = fel
        if fel <= 0:
            self.__terminated = True
        return f

    def register(self, x, f: Union[int, float]) -> None:
        """
        Register that an objective function evaluation has been performed.

        :param x: the solution
        :param f: the corresponding objective value
        """
        self.__register(x, f)
        fel: Final[int] = self.__fes_left - 1
        self.__fes_left = fel
        if fel <= 0:
            self.__terminated = True

    def get_consumed_fes(self) -> int:
        """
        Get the number of consumed FEs.

        :returns: the number of consumed FEs
        """
        return self.max_fes - self.__fes_left

    def get_last_improvement_fe(self) -> int:
        """
        Get the FE index when the last improvement happened.

        :returns: the FE index when the last improvement happened
        """
        return max(1 if self.__fes_left < self.max_fes else 0,
                   self._owner.get_last_improvement_fe() - self.__start_fe)

    def get_max_fes(self) -> int:
        """
        Get the maximum FEs.

        :returns: the maximum FEs of this process slice.
        """
        return self.max_fes

    def __str__(self) -> str:
        """
        Get the name of this subprocess implementation.

        :return: "forFEs_{max_fes}_{owner}"
        """
        return f"forFEs_{self.max_fes}_{self._owner}"


class FromStatingPointForFEs(Process):
    """A process searching from a given point for a fixed amount of FEs."""

    def __init__(self, owner: Process,
                 in_and_out: Any,
                 f: Union[int, float],
                 max_fes: int):
        """
        Create a sub-process searching for some FEs from one starting point.

        :param owner: the owning process
        :param in_and_out: the input solution record, which will be
            overwritten with the best encountered solution
        :param f: the objective value corresponding to `in_and_out`
        :param max_fes: the maximum number of FEs granted to the sub-process
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
        #: the maximum FEs
        self.max_fes: Final[int] = check_max_fes(max_fes)
        #: the FEs that we still have left
        self.__fes_left: int = max_fes
        #: did we terminate?
        self.__terminated: bool = False
        #: the best solution
        self.__best_x: Final[Any] = in_and_out
        #: the best-so-far solution
        self.__best_f: Union[int, float] = f
        #: the last improvement fe
        self.__last_improvement_fe: int = 0
        #: the fast call to the owner's shouldTerminate method
        self.__should_terminate: Final[Callable[[], bool]] \
            = owner.should_terminate
        #: the fast call to the owner's evaluate method
        self.__evaluate: Final[Callable[[Any], Union[int, float]]] \
            = owner.evaluate
        #: the fast call to the owner's register method
        self.__register: Final[Callable[[Any, Union[int, float]], None]] \
            = owner.register

    def should_terminate(self) -> bool:
        """
        Check if this sub-process should terminate.

        :returns: `True` if the allotted FEs have been consumed
        """
        return self.__terminated or self.__should_terminate()

    def terminate(self) -> None:
        """Terminate this sub-process."""
        self.__terminated = True

    def has_best(self) -> bool:
        """
        Check whether we have a current-best solution.

        :returns: `True`
        """
        return True

    def get_copy_of_best_x(self, x) -> None:
        """
        Get a copy of the current-best solution.

        :param x: the container to receive the copy
        """
        self.copy(x, self.__best_x)

    def get_best_f(self) -> Union[int, float]:
        """
        Get the best-so-far objective value.

        :returns: the best-so-far objective value
        """
        return self.__best_f

    def evaluate(self, x) -> Union[float, int]:
        """
        Evaluate a solution and return the objective value.

        :param x: the solution
        :returns: the corresponding objective value
        """
        f: Final[Union[int, float]] = self.__evaluate(x)
        fel: Final[int] = self.__fes_left - 1
        self.__fes_left = fel
        if f < self.__best_f:
            self.__best_f = f
            self.copy(self.__best_x, x)
            self.__last_improvement_fe = self.max_fes - fel
        if fel <= 0:
            self.__terminated = True
        return f

    def register(self, x, f: Union[int, float]) -> None:
        """
        Register that an objective function evaluation has been performed.

        :param x: the solution
        :param f: the corresponding objective value
        """
        self.__register(x, f)
        fel: Final[int] = self.__fes_left - 1
        self.__fes_left = fel
        if f < self.__best_f:
            self.__best_f = f
            self.copy(self.__best_x, x)
            self.__last_improvement_fe = self.max_fes - fel
        if fel <= 0:
            self.__terminated = True

    def get_consumed_fes(self) -> int:
        """
        Get the number of consumed FEs.

        :returns: the number of consumed FEs
        """
        return self.max_fes - self.__fes_left

    def get_last_improvement_fe(self) -> int:
        """
        Get the FE index when the last improvement happened.

        :returns: the FE index when the last improvement happened
        """
        return self.__last_improvement_fe

    def get_max_fes(self) -> int:
        """
        Get the maximum FEs.

        :returns: the maximum FEs of this process slice.
        """
        return self.max_fes

    def __str__(self) -> str:
        """
        Get the name of this subprocess implementation.

        :return: "fromStartForFEs_{max_fes}_{owner}"
        """
        return f"fromStartForFEs_{self.max_fes}_{self._owner}"
