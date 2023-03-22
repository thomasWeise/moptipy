"""Test the space of permutations of selections."""
from math import factorial

import numpy as np
from numpy.random import default_rng

# noinspection PyPackageRequirements
from pytest import raises

from moptipy.spaces.ordered_choices import OrderedChoices
from moptipy.tests.on_ordered_choices import (
    choices_for_tests,
    make_choices_valid,
)
from moptipy.tests.space import validate_space
from moptipy.utils.logger import InMemoryLogger


def test_signed_permutations() -> None:
    """Test the signed permutations."""
    def _invalid(x) -> np.ndarray:
        x[0] = 33
        return x

    space = OrderedChoices.signed_permutations(10)
    assert space.min_value == -10
    assert space.max_value == 10
    assert space.dimension == 10
    assert all(space.blueprint == np.array(list(range(-10, 0, 1))))
    assert space.n_points() == factorial(10) * (2 ** 10)
    assert space.to_str(space.create()) == "-10;-9;-8;-7;-6;-5;-4;-3;-2;-1"
    assert str(space) == "selperm10"
    assert space.is_compatible_with_permutations()
    validate_space(space, make_element_invalid=_invalid)

    with InMemoryLogger() as log:
        with log.key_values("F") as kv:
            space.log_parameters_to(kv)
        result = log.get_log()
    assert result == [
        "BEGIN_F",
        "name: selperm10",
        "class: moptipy.spaces.ordered_choices.OrderedChoices",
        "nvars: 10",
        "dtype: b",
        "min: -10",
        "max: 10",
        "choices: -10;10/-9;9/-8;8/-7;7/-6;6/-5;5/-4;4/-3;3/-2;2/-1;1",
        "END_F"]

    space = OrderedChoices.signed_permutations(1)
    assert space.min_value == -1
    assert space.max_value == 1
    assert space.dimension == 1
    assert str(space) == "selperm1"
    assert all(space.blueprint == np.array(list(range(-1, 0, 1))))
    assert space.n_points() == 2
    assert not space.is_compatible_with_permutations()
    assert space.to_str(space.create()) == "-1"
    validate_space(space, make_element_invalid=_invalid)
    with InMemoryLogger() as log:
        with log.key_values("F") as kv:
            space.log_parameters_to(kv)
        result = log.get_log()
    assert result == [
        "BEGIN_F",
        "name: selperm1",
        "class: moptipy.spaces.ordered_choices.OrderedChoices",
        "nvars: 1",
        "dtype: b",
        "min: -1",
        "max: 1",
        "choices: -1;1",
        "END_F"]


def test_ordered_choices() -> None:
    """Test the ordered choices for tests."""
    random = default_rng()
    for choices in choices_for_tests():
        assert isinstance(choices, OrderedChoices)
        assert choices.n_points() > 1
        base = choices.create()
        assert isinstance(base, np.ndarray)
        assert len(base) == choices.dimension
        choices.validate(base)

        forbidden = {(x for x in y) for y in choices.choices.values()}
        use_impossible = None
        for k in range(np.iinfo(choices.dtype).min,
                       np.iinfo(choices.dtype).max):
            if k not in forbidden:
                use_impossible = int(k)
        assert isinstance(use_impossible, int)

        def _invalid(x, ui=use_impossible) -> np.ndarray:
            x[0] = ui
            return x

        def _valid(x, v=make_choices_valid(choices), g=random) -> np.ndarray:
            v(g, x)
            return x

        validate_space(choices, make_element_invalid=_invalid,
                       make_element_valid=_valid)


def test_invalid_ordered_choices() -> None:
    """Check that invalid ordered choices correctly throw errors."""
    with raises(TypeError):
        OrderedChoices("123")

    with raises(TypeError):
        OrderedChoices([[1, 2], ["x"]])

    with raises(ValueError):
        OrderedChoices([[1]])

    with raises(ValueError):
        OrderedChoices([[1], [2], [3, 1]])

    with raises(ValueError):
        OrderedChoices([[1], [1]])

    OrderedChoices([[1], [1], [2]])
    OrderedChoices([[1], [1], [2], [2]])

    with raises(ValueError):
        OrderedChoices([[1], [1], [2], [2, 3]])
