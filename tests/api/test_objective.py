from moptipy.api import CallableObjective, Objective, Component
from math import inf
from pytest import raises
from moptipy.utils import TempFile, Logger


def test_pure_callable_objective_function():
    f = CallableObjective(lambda x: 5)
    assert isinstance(f, Objective)
    assert isinstance(f, Component)

    assert f.evaluate(7) == 5
    assert str(f) == "unnamed_function"
    assert f.get_name() == "unnamed_function"
    assert f.upper_bound() == +inf
    assert f.lower_bound() == -inf

    with raises(ValueError):
        CallableObjective(None)

    with raises(ValueError):
        CallableObjective("blabla")


def test_callable_objective_function_bounds():
    f = CallableObjective(lambda x: -3, lower_bound=7)
    assert isinstance(f, Objective)
    assert isinstance(f, Component)
    assert f.evaluate(7) == -3
    assert str(f) == "unnamed_function"
    assert f.get_name() == "unnamed_function"
    assert f.upper_bound() == +inf
    assert f.lower_bound() == 7

    f = CallableObjective(lambda x: -3, upper_bound=7)
    assert isinstance(f, Objective)
    assert isinstance(f, Component)
    assert f.evaluate(7) == -3
    assert str(f) == "unnamed_function"
    assert f.get_name() == "unnamed_function"
    assert f.upper_bound() == 7
    assert f.lower_bound() == -inf

    f = CallableObjective(lambda x: -3, upper_bound=7, lower_bound=-4)
    assert isinstance(f, Objective)
    assert isinstance(f, Component)
    assert f.evaluate(7) == -3
    assert str(f) == "unnamed_function"
    assert f.get_name() == "unnamed_function"
    assert f.upper_bound() == 7
    assert f.lower_bound() == -4

    with raises(ValueError):
        CallableObjective(lambda x: -3, upper_bound=4, lower_bound=4)

    with raises(ValueError):
        CallableObjective(lambda x: -3, upper_bound=-3, lower_bound=4)

    with raises(ValueError):
        CallableObjective(lambda x: -3, lower_bound="x")

    with raises(ValueError):
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

    with raises(ValueError):
        CallableObjective(lambda x: -3, name=" ")


def test_logged_args():
    f = CallableObjective(lambda x: 5,
                          lower_bound=-5,
                          upper_bound=11,
                          name="hello")
    assert isinstance(f, Objective)
    assert isinstance(f, Component)

    with TempFile() as tmp:
        path = str(tmp)
        with Logger(path) as log:
            with log.key_values("F") as kv:
                f.log_parameters_to(kv)
        result = open(path, "r").read().splitlines()
    assert result == ["BEGIN_F",
                      "name:hello",
                      "type:<class 'moptipy.api.objective.CallableObjective'>",
                      "innerType:<class 'function'>",
                      "lowerBound:-5",
                      "upperBound:11",
                      "END_F"]
