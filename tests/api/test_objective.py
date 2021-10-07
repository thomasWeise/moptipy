"""Test the callable objective function."""
from math import inf

# noinspection PyPackageRequirements
from pytest import raises

from moptipy.api import CallableObjective, Objective, Component
from moptipy.tests.objective import check_objective
from moptipy.utils.temp import TempFile
from moptipy.utils.logger import FileLogger


def test_pure_callable_objective_function():
    f = CallableObjective(lambda x: 5)
    assert isinstance(f, Objective)
    assert isinstance(f, Component)

    assert f.evaluate(7) == 5
    assert str(f) == "unnamed_function"
    assert f.get_name() == "unnamed_function"
    assert f.upper_bound() == +inf
    assert f.lower_bound() == -inf

    with raises(TypeError):
        # noinspection PyTypeChecker
        CallableObjective(None)

    with raises(TypeError):
        # noinspection PyTypeChecker
        CallableObjective("blabla")

    check_objective(f, lambda: 3)


def test_callable_objective_function_bounds():
    f = CallableObjective(lambda x: 13, lower_bound=7)
    assert isinstance(f, Objective)
    assert isinstance(f, Component)
    assert f.evaluate(7) == 13
    assert str(f) == "unnamed_function"
    assert f.get_name() == "unnamed_function"
    assert f.upper_bound() == +inf
    assert f.lower_bound() == 7
    check_objective(f, lambda: 3)

    f = CallableObjective(lambda x: 3, upper_bound=7)
    assert isinstance(f, Objective)
    assert isinstance(f, Component)
    assert f.evaluate(7) == 3
    assert str(f) == "unnamed_function"
    assert f.get_name() == "unnamed_function"
    assert f.upper_bound() == 7
    assert f.lower_bound() == -inf
    check_objective(f, lambda: 3)

    f = CallableObjective(lambda x: -3, upper_bound=7, lower_bound=-4)
    assert isinstance(f, Objective)
    assert isinstance(f, Component)
    assert f.evaluate(7) == -3
    assert str(f) == "unnamed_function"
    assert f.get_name() == "unnamed_function"
    assert f.upper_bound() == 7
    assert f.lower_bound() == -4
    check_objective(f, lambda: 3)

    with raises(ValueError):
        CallableObjective(lambda x: -3, upper_bound=4, lower_bound=4)

    with raises(ValueError):
        CallableObjective(lambda x: -3, upper_bound=-3, lower_bound=4)

    with raises(TypeError):
        # noinspection PyTypeChecker
        CallableObjective(lambda x: -3, lower_bound="x")

    with raises(TypeError):
        # noinspection PyTypeChecker
        CallableObjective(lambda x: -3, upper_bound="x")


def test_named_callable_objective_function():
    f = CallableObjective(lambda x: -3, name="hallo")
    assert str(f) == "hallo"
    assert f.get_name() == "hallo"
    f = CallableObjective(lambda x: -3, name=" hallo")
    assert str(f) == "hallo"
    assert f.get_name() == "hallo"
    f = CallableObjective(lambda x: -3, name="hallo ")
    assert str(f) == "hallo"
    assert f.get_name() == "hallo"
    check_objective(f, lambda: 3)

    with raises(ValueError):
        CallableObjective(lambda x: -3, name=" ")


def test_logged_args():
    f = CallableObjective(lambda x: 5,
                          lower_bound=-5,
                          upper_bound=11,
                          name="hello")
    assert isinstance(f, Objective)
    assert isinstance(f, Component)

    with TempFile.create() as path:
        with FileLogger(path) as log:
            with log.key_values("F") as kv:
                f.log_parameters_to(kv)
        result = path.read_all_list()
    assert result == [
        "BEGIN_F",
        "name: hello",
        "type: <class 'moptipy.api.objective.CallableObjective'>",
        "innerType: <class 'function'>",
        "lowerBound: -5",
        "upperBound: 11",
        "END_F"]
