"""A set of process wrappers that can be used for surrogate modeling."""

from math import inf, isfinite
from typing import Any, Callable, Final

import numpy as np

from moptipy.api.process import Process


class _Surrogate(Process):
    """A surrogate process."""

    def __init__(self, owner: Process, max_fes: int):
        super().__init__()
        #: the owning process
        self._owner: Final[Process] = owner
        self.get_random = owner.get_random  # type: ignore
        a = owner.get_consumed_time_millis  # type: ignore
        self.get_consumed_time_millis = a  # type: ignore
        a = owner.get_max_time_millis  # type: ignore
        self.get_max_time_millis = a  # type: ignore
        a = owner.get_last_improvement_time_millis  # type: ignore
        self.get_last_improvement_time_millis = a  # type: ignore
        self.has_log = owner.has_log  # type: ignore
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
        self.max_fes: Final[int] = max_fes
        #: the FEs that we still have left
        self._fes_left: int = max_fes
        #: did we terminate?
        self._terminated: bool = False
        #: the fast call to the owner's should_terminate method
        self.__should_terminate: Final[Callable[[], bool]] \
            = owner.should_terminate

    def should_terminate(self) -> bool:
        return self._terminated or self.__should_terminate()

    def terminate(self) -> None:
        self._terminated = True

    def register(self, x, f: int | float) -> None:
        raise ValueError("you should not call this function.")

    def get_consumed_fes(self) -> int:
        return self.max_fes - self._fes_left

    def get_last_improvement_fe(self) -> int:
        return 1 if self._fes_left < self.max_fes else 0

    def get_max_fes(self) -> int:
        return self.max_fes

    def __str__(self) -> str:
        return f"{self.max_fes}_{self._owner}"


class _SurrogateApply(_Surrogate):
    """A process running for a `n` FEs and collecting the results."""

    #: the internal evaluation function
    _evaluate: Callable[[np.ndarray], np.ndarray]

    def __init__(self, owner: Process, max_fes: int) -> None:
        super().__init__(owner, max_fes)
        #: the best-so-far solution
        self._best_x: Final[np.ndarray] = owner.create()
        #: the best-so-far objective value
        self._best_f: int | float = inf

    def has_best(self) -> bool:
        return isfinite(self._best_f)

    def get_best_f(self) -> int | float:
        return self._best_f

    def get_copy_of_best_x(self, x) -> None:
        np.copyto(x, self._best_x)

    def evaluate(self, x) -> float | int:
        f: Final[float] = self._evaluate(
            x.reshape((1, x.shape[0])))[0]
        fel: Final[int] = self._fes_left - 1
        self._fes_left = fel
        if fel <= 0:
            self._terminated = True
        if isfinite(f):
            if f < self._best_f:
                self._best_f = f
                np.copyto(self._best_x, x)
            return f
        return inf


class _SurrogateWarmup(_Surrogate):
    """A process running for a `n` FEs and collecting the results."""

    def __init__(self, owner: Process, max_fes: int,
                 point_collector: Callable[[np.ndarray], None],
                 f_collector: Callable[[int | float], None]):
        super().__init__(owner, max_fes)
        #: the point collector
        self.__point_collector = point_collector
        #: the objective value collector
        self.__f_collector = f_collector
        self.has_best = owner.has_best  # type: ignore
        self.get_copy_of_best_x = owner.get_copy_of_best_x  # type: ignore
        self.get_best_f = owner.get_best_f  # type: ignore
        #: the owner's evaluation function
        self.__evaluate: Final[Callable[[Any], int | float]] = owner.evaluate

    def evaluate(self, x) -> float | int:
        f: Final[int | float] = self.__evaluate(x)
        self.__point_collector(x.copy())
        self.__f_collector(f)
        fel: Final[int] = self._fes_left - 1
        self._fes_left = fel
        if fel <= 0:
            self._terminated = True
        return f

    def lower_bound(self) -> float:
        return -inf

    def upper_bound(self) -> float:
        return inf

    def __str__(self) -> str:
        return f"surrogateWarmup_{super().__str__()}"
