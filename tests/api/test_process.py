"""Test the Process API."""
from math import sin
from os.path import getsize, isfile
from typing import Final

import numpy as np

# noinspection PyPackageRequirements
import pytest
from numpy.random import Generator

from moptipy.api import logging
from moptipy.api.algorithm import Algorithm
from moptipy.api.execution import Execution
from moptipy.api.objective import Objective
from moptipy.api.process import Process
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils import logger
from moptipy.utils.temp import TempFile

#: the initial worst objective value
WORST_F: Final[int] = 1_000_000_000_000
#: the number of FEs
MAX_FES: Final[int] = 100
#: the best objective value after MAX_FES
BEST_F: Final[int] = WORST_F - MAX_FES + 1


class MyAlgorithm1(Algorithm):
    """The algorithm for testing."""

    def solve(self, process: Process) -> None:
        """Perform a simple algorithm."""
        x = process.create()
        assert isinstance(x, np.ndarray)
        g = process.get_random()
        assert isinstance(g, Generator)
        assert not process.has_best()

        i = WORST_F
        while not process.should_terminate():
            x.fill(i)
            process.evaluate(x)
            assert process.has_best()
            i -= 1
        assert all(x < WORST_F)


class MyAlgorithm2(Algorithm):
    """The second algorithm for testing."""

    def solve(self, process: Process) -> None:
        """Conduct a second algorithm simply iterating over x."""
        x = process.create()
        assert isinstance(x, np.ndarray)
        g = process.get_random()
        assert isinstance(g, Generator)
        assert not process.has_best()

        i = 1000
        while not process.should_terminate():
            x.fill(i)
            process.evaluate(x)
            i -= 1
            assert process.has_best()


class MyObjective1(Objective):
    """The internal test objective."""

    def evaluate(self, x) -> float | int:
        """Return x[0] as dummy objective value."""
        return x[0]


class MyObjective2(Objective):
    """The internal test objective."""

    def evaluate(self, x) -> float | int:
        """Return x[0] as dummy objective value."""
        return sin(x[0])


def test_process_noss_no_log() -> None:
    """Test processes without search space and without log."""
    v = VectorSpace(10, -1e100, 1e100)
    x = v.create()
    exp = Execution()
    exp.set_algorithm(MyAlgorithm1())
    exp.set_solution_space(v)
    exp.set_objective(MyObjective1())
    exp.set_max_fes(MAX_FES)
    with exp.execute() as p:
        assert p.has_best()
        assert p.get_best_f() == BEST_F
        p.get_copy_of_best_x(x)
        assert all(x == BEST_F)
        p.get_copy_of_best_y(x)
        assert all(x == BEST_F)


def test_process_noss_log() -> None:
    """Test processes without search space and with log."""
    v = VectorSpace(10, -1e100, 1e100)
    x = v.create()

    with TempFile.create() as path:
        exp = Execution()
        exp.set_algorithm(MyAlgorithm1())
        exp.set_solution_space(v)
        exp.set_objective(MyObjective1())
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
            result = file.read().splitlines()
        assert len(result) > 5


def test_process_noss_timed_log() -> None:
    """Test processes without search space, with log, with time limit."""
    v = VectorSpace(10, -1e100, 1e100)
    x = v.create()
    with TempFile.create() as path:
        exp = Execution()
        exp.set_algorithm(MyAlgorithm1())
        exp.set_solution_space(v)
        exp.set_objective(MyObjective1())
        exp.set_log_file(path)
        exp.set_max_time_millis(20)
        with exp.execute() as p:
            assert p.has_best()
            lll = p.get_best_f()
            assert lll < WORST_F
            p.get_copy_of_best_x(x)
            assert all(x == lll)
            p.get_copy_of_best_y(x)
            assert all(x == lll)
        assert isfile(path)
        assert getsize(path) > 10
        with open(path) as file:
            result = file.read().splitlines()
        assert len(result) > 5


def test_process_noss_maxfes_log_state() -> None:
    """Test processes without search space, with log, and with MAX_FES."""
    v = VectorSpace(4, -1e100, 1e100)
    with TempFile.create() as path:
        exp = Execution()
        exp.set_algorithm(MyAlgorithm2())
        exp.set_solution_space(v)
        exp.set_objective(MyObjective2())
        exp.set_log_file(path)
        exp.set_max_fes(500)
        exp.set_log_improvements()
        with exp.execute() as p:
            assert p.has_best()
        assert isfile(path)
        assert getsize(path) > 10
        result = path.read_all_list()
        assert len(result) > 5


class MyAlgorithm3(Algorithm):
    """The third algorithm for testing."""

    def solve(self, process: Process) -> None:
        """Perform an algorithm that throws an error."""
        x = process.create()
        assert isinstance(x, np.ndarray)
        g = process.get_random()
        assert isinstance(g, Generator)
        assert not process.has_best()

        i = 1000
        while not process.should_terminate():
            x.fill(i)
            process.evaluate(x)
            i -= 1
            assert process.has_best()
            raise ValueError("Haha!")


def test_process_with_error() -> None:
    """Test processes that throws an error."""
    v = VectorSpace(4, -1e100, 1e100)
    with TempFile.create() as path:
        exp = Execution()
        exp.set_algorithm(MyAlgorithm3())
        exp.set_solution_space(v)
        exp.set_objective(MyObjective2())
        exp.set_log_file(path)
        exp.set_max_fes(500)
        exp.set_log_improvements()
        with pytest.raises(ValueError):
            with exp.execute() as p:
                assert p.has_best()
            assert isfile(path)
            assert getsize(path) > 10
            result = path.read_all_list()
            assert len(result) > 5
            assert (logger.SECTION_START
                    + logging.SECTION_ERROR_IN_RUN) in result
