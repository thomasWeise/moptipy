"""Test the Logging Process API."""
from os.path import getsize, isfile
from typing import Final

import numpy as np
from numpy.random import Generator

from moptipy.api.algorithm import Algorithm
from moptipy.api.execution import Execution
from moptipy.api.objective import Objective
from moptipy.api.process import Process
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.temp import TempFile

#: the initial worst objective value
WORST_F: Final[int] = 1_000_000_000_000
#: the number of FEs
MAX_FES: Final[int] = 100
#: the best objective value after MAX_FES
BEST_F: Final[int] = WORST_F - MAX_FES + 1


class MyAlgorithm(Algorithm):
    """The first algorithm for testing."""

    def solve(self, process: Process) -> None:
        """Perform a simple algorithm."""
        x = process.create()
        assert isinstance(x, np.ndarray)
        g = process.get_random()
        assert isinstance(g, Generator)
        assert not process.has_best()

        process.add_log_section("B", "Y\nZ\n")
        i = WORST_F
        while not process.should_terminate():
            x.fill(i)
            process.evaluate(x)
            assert process.has_best()
            i -= 1
        assert all(x < WORST_F)
        process.add_log_section("A", "X")


class MyObjective(Objective):
    """The internal test objective."""

    def evaluate(self, x) -> float | int:
        """Return x[0] as dummy objective value."""
        return x[0]


def test_process_noss_log() -> None:
    """Test processes without search space and with log."""
    v = VectorSpace(10, -1e100, 1e100)
    x = v.create()

    with TempFile.create() as path:
        exp = Execution()
        exp.set_algorithm(MyAlgorithm())
        exp.set_solution_space(v)
        exp.set_objective(MyObjective())
        exp.set_log_file(path)
        exp.set_max_fes(MAX_FES)
        with exp.execute() as p:
            assert p.has_best()
            assert p.get_best_f() == BEST_F
            p.get_copy_of_best_x(x)
            assert all(x == BEST_F)
            p.get_copy_of_best_y(x)
            assert all(x == BEST_F)
        assert isfile(path)
        assert getsize(path) > 10
        with open(path) as file:
            result = [line.rstrip() for line in file.read().splitlines()]
        assert len(result) > 5

        i = result.index("BEGIN_A")
        i += 1
        assert result[i] == "X"
        i += 1
        assert result[i] == "END_A"
        i += 1
        assert result[i] == "BEGIN_B"
        i += 1
        assert result[i] == "Y"
        i += 1
        assert result[i] == "Z"
        i += 1
        assert result[i] == "END_B"
        i += 1
        assert i == len(result)
