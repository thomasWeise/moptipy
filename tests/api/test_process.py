"""Test the Process API."""
from math import sin
from os.path import isfile, getsize

import numpy as np
from numpy.random import Generator

from moptipy.api import CallableAlgorithm, CallableObjective, \
    Process
from moptipy.api.execution import Execution
from moptipy.spaces import VectorSpace
from moptipy.utils.temp import TempFile

worst_f = 1_000_000_000_000
max_fes = 100
best_f = worst_f - max_fes + 1


def myalgorithm(process: Process):
    x = process.create()
    assert isinstance(x, np.ndarray)
    g = process.get_random()
    assert isinstance(g, Generator)
    assert not process.has_current_best()

    i = worst_f
    while not process.should_terminate():
        x.fill(i)
        process.evaluate(x)
        i -= 1
        assert process.has_current_best()
    assert all(x < worst_f)


def myalgorithm2(process: Process):
    x = process.create()
    assert isinstance(x, np.ndarray)
    g = process.get_random()
    assert isinstance(g, Generator)
    assert not process.has_current_best()

    i = 1000
    while not process.should_terminate():
        x.fill(i)
        process.evaluate(x)
        i -= 1
        assert process.has_current_best()


def myobjective(x):
    return x[0]


def myobjective2(x):
    return sin(x[0])


def test_process_noss_no_log():
    v = VectorSpace(10)
    x = v.create()
    exp = Execution()
    exp.set_algorithm(CallableAlgorithm(myalgorithm))
    exp.set_solution_space(v)
    exp.set_objective(CallableObjective(myobjective))
    exp.set_max_fes(max_fes)
    with exp.execute() as p:
        assert p.has_current_best()
        assert p.get_current_best_f() == best_f
        p.get_copy_of_current_best_x(x)
        assert all(x == best_f)
        p.get_copy_of_current_best_y(x)
        assert all(x == best_f)


def test_process_noss_log():
    v = VectorSpace(10)
    x = v.create()
    with TempFile.create() as path:
        exp = Execution()
        exp.set_algorithm(CallableAlgorithm(myalgorithm))
        exp.set_solution_space(v)
        exp.set_objective(CallableObjective(myobjective))
        exp.set_log_file(path)
        exp.set_max_fes(max_fes)
        with exp.execute() as p:
            assert p.has_current_best()
            assert p.get_current_best_f() == best_f
            p.get_copy_of_current_best_x(x)
            assert all(x == best_f)
            p.get_copy_of_current_best_y(x)
            assert all(x == best_f)
        assert isfile(path)
        assert getsize(path) > 10
        result = open(path, "r").read().splitlines()
        assert len(result) > 5


def test_process_noss_timed_log():
    v = VectorSpace(10)
    x = v.create()
    with TempFile.create() as path:
        exp = Execution()
        exp.set_algorithm(CallableAlgorithm(myalgorithm))
        exp.set_solution_space(v)
        exp.set_objective(CallableObjective(myobjective))
        exp.set_log_file(path)
        exp.set_max_time_millis(20)
        with exp.execute() as p:
            assert p.has_current_best()
            lll = p.get_current_best_f()
            assert lll < worst_f
            p.get_copy_of_current_best_x(x)
            assert all(x == lll)
            p.get_copy_of_current_best_y(x)
            assert all(x == lll)
        assert isfile(path)
        assert getsize(path) > 10
        result = open(path, "r").read().splitlines()
        assert len(result) > 5


def test_process_noss_maxfes_log_state():
    v = VectorSpace(4)
    with TempFile.create() as path:
        exp = Execution()
        exp.set_algorithm(CallableAlgorithm(myalgorithm2))
        exp.set_solution_space(v)
        exp.set_objective(CallableObjective(myobjective2))
        exp.set_log_file(path)
        exp.set_max_fes(500)
        exp.set_log_improvements()
        with exp.execute() as p:
            assert p.has_current_best()
        assert isfile(path)
        assert getsize(path) > 10
        result = path.read_all_list()
        assert len(result) > 5
