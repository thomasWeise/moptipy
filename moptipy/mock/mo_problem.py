"""The mock multi-objective optimization problem."""

from typing import Iterable, cast

import numpy as np
from numpy.random import default_rng
from pycommons.types import check_int_range, type_error

from moptipy.mo.problem.weighted_sum import WeightedSum
from moptipy.mock.objective import MockObjective


class MockMOProblem(WeightedSum):
    """A mock-up of a multi-objective optimization problem."""

    def __init__(self, objectives: Iterable[MockObjective],
                 weights: Iterable[int | float]) -> None:
        """
        Create the mock multi-objective problem.

        :param objectives: the mock objectives
        :param weights: their weights
        """
        super().__init__(objectives, weights)

    @staticmethod
    def for_dtype(n: int, dtype: np.dtype) -> "MockMOProblem":
        """
        Create a mock multi-objective problem.

        :param n: the number of objectives
        :param dtype: the optional dtype
        :returns: the mock multi-objective problem
        """
        check_int_range(n, "n", 1, 1_000)
        if not isinstance(dtype, np.dtype):
            raise type_error(dtype, "dtype", np.dtype)

        random = default_rng()
        weights: list[int | float] =\
            [int(w) for w in random.integers(1, 3, n)] \
            if random.integers(2) <= 0 \
            else [float(w) for w in random.uniform(0.2, 3, n)]
        max_trials: int = 1000
        while max_trials > 0:
            max_trials -= 1
            funcs = [MockObjective.for_type(dtype) for _ in range(n)]
            names = {str(o) for o in funcs}
            if len(names) >= n:
                return MockMOProblem(funcs, weights)
        raise ValueError("Huh?")

    def get_objectives(self) -> tuple[MockObjective, ...]:
        """
        Get the internal objective functions.

        :return: the internal mock objective functions
        """
        return cast(tuple[MockObjective, ...], self._objectives)

    def sample(self, fs: np.ndarray) -> int | float:
        """
        Sample one vector of ojective values.

        :param fs: the array to receive the objective values
        :returns: the scalarized objective values
        """
        for i, o in enumerate(self._objectives):
            fs[i] = cast(MockObjective, o).sample()
        return self._scalarize(fs)
