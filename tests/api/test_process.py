import numpy as np
from numpy.random import Generator

from moptipy.api import Algorithm, CallableAlgorithm, CallableObjective,\
    Process
from moptipy.spaces import VectorSpace
from moptipy.utils import TempFile
from os.path import isfile, getsize
from math import sin, log2

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

    process.log_state("beta", 23)

    i = 1000
    while not process.should_terminate():
        x.fill(i)
        process.evaluate(x)
        i -= 1
        assert process.has_current_best()
        if i % 3 == 0:
            process.log_state("alpha", sin(i))
        if i % 12 == 0:
            process.log_state("beta", log2(i + sin(i)))
        if i % 95 == 0:
            process.log_state("gamma", sin(9 * log2(i + sin(i))))


def myobjective(x):
    return x[0]


def myobjective2(x):
    return sin(x[0])


def test_process_noss_no_log():
    v = VectorSpace(10)
    x = v.create()
    with Algorithm.apply(algorithm=CallableAlgorithm(myalgorithm),
                         solution_space=v,
                         objective=CallableObjective(myobjective),
                         max_fes=max_fes) as p:
        assert p.has_current_best()
        assert p.get_current_best_f() == best_f
        p.get_copy_of_current_best_x(x)
        assert all(x == best_f)
        p.get_copy_of_current_best_y(x)
        assert all(x == best_f)


def test_process_noss_log():
    v = VectorSpace(10)
    x = v.create()
    with TempFile() as log:
        path = str(log)
        with Algorithm.apply(algorithm=CallableAlgorithm(myalgorithm),
                             solution_space=v,
                             objective=CallableObjective(myobjective),
                             max_fes=max_fes,
                             log_file=path,
                             overwrite_log=True) as p:
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
    with TempFile() as log:
        path = str(log)
        with Algorithm.apply(algorithm=CallableAlgorithm(myalgorithm),
                             solution_space=v,
                             objective=CallableObjective(myobjective),
                             max_time_millis=20,
                             log_file=path,
                             overwrite_log=True) as p:
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
    with TempFile() as log:
        path = str(log)
        with Algorithm.apply(algorithm=CallableAlgorithm(myalgorithm2),
                             solution_space=v,
                             objective=CallableObjective(myobjective2),
                             max_fes=500,
                             log_file=path,
                             log_improvements=True,
                             overwrite_log=True) as p:
            assert p.has_current_best()
        assert isfile(path)
        assert getsize(path) > 10
        result = open(path, "r").read().splitlines()
        assert len(result) > 5
