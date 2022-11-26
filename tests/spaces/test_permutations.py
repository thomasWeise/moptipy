"""Test the space of permutations."""
from math import factorial

import numpy as np

from moptipy.spaces.permutations import Permutations
from moptipy.tests.space import validate_space
from moptipy.utils.logger import InMemoryLogger


def test_permutations_with_repetitions() -> None:
    """Test the permutations with repetitions."""
    def _invalid(x) -> np.ndarray:
        x[0] = 33
        return x

    space = Permutations.with_repetitions(10, 1)
    assert space.min_value == 0
    assert space.max_value == 9
    assert space.dimension == 10
    assert all(space.blueprint == np.array(list(range(10))))
    assert space.n_points() == factorial(10)
    assert space.to_str(space.create()) == "0;1;2;3;4;5;6;7;8;9"
    assert str(space) == "perm10"
    assert space.is_dense()
    validate_space(space, make_element_invalid=_invalid)

    space = Permutations.with_repetitions(2, 1)
    assert space.min_value == 0
    assert space.max_value == 1
    assert space.dimension == 2
    assert str(space) == "perm2"
    assert all(space.blueprint == np.array(list(range(2))))
    assert space.n_points() == 2
    assert space.is_dense()
    assert space.to_str(space.create()) == "0;1"
    validate_space(space, make_element_invalid=_invalid)

    space = Permutations.with_repetitions(2, 12)
    assert space.min_value == 0
    assert space.max_value == 1
    assert space.dimension == 2 * 12
    assert str(space) == "perm2w12r"
    assert all(space.blueprint == np.array(([0] * 12) + ([1] * 12)))
    assert space.n_points() == (factorial(2 * 12) // factorial(12)) \
           // factorial(12)
    assert space.is_dense()
    assert space.to_str(space.create()) == \
           "0;0;0;0;0;0;0;0;0;0;0;0;1;1;1;1;1;1;1;1;1;1;1;1"
    validate_space(space, make_element_invalid=_invalid)

    space = Permutations.with_repetitions(21, 11)
    assert space.min_value == 0
    assert space.max_value == 20
    assert space.dimension == 21 * 11
    assert str(space) == "perm21w11r"
    assert all(space.blueprint == np.array(sorted(list(range(21)) * 11)))
    assert space.is_dense()
    validate_space(space, make_element_invalid=_invalid)


def test_standard_permutations() -> None:
    """Test the permutation space."""
    def _invalid(x) -> np.ndarray:
        x[0] = 111
        return x

    space = Permutations.standard(12)
    assert space.min_value == 0
    assert space.max_value == 11
    assert space.dimension == 12
    assert all(space.blueprint == np.array(list(range(12))))
    assert space.n_points() == factorial(12)
    assert str(space) == "perm12"
    assert space.is_dense()

    validate_space(space, make_element_invalid=_invalid)

    space = Permutations.standard(2)
    assert space.min_value == 0
    assert space.max_value == 1
    assert space.dimension == 2
    assert space.n_points() == 2
    assert all(space.blueprint == np.array(list(range(2))))
    assert str(space) == "perm2"
    assert space.is_dense()

    validate_space(space, make_element_invalid=_invalid)


def test_permutations_blueprint() -> None:
    """Test the permutation space."""
    def _invalid(x) -> np.ndarray:
        x[0] = 111
        return x

    space = Permutations([4, 6, 12, 5])
    assert space.min_value == 4
    assert space.max_value == 12
    assert space.dimension == 4
    assert str(space) == "permOfString"
    assert all(space.blueprint == np.array([4, 5, 6, 12]))
    assert not space.is_dense()
    assert space.n_points() == factorial(4)
    assert space.to_str(space.create()) == "4;5;6;12"

    with InMemoryLogger() as log:
        with log.key_values("F") as kv:
            space.log_parameters_to(kv)
        result = log.get_log()
    assert result == ["BEGIN_F",
                      "name: permOfString",
                      "class: moptipy.spaces.permutations.Permutations",
                      "nvars: 4",
                      "dtype: b",
                      "min: 4",
                      "max: 12",
                      "repetitions: 1",
                      "baseString: 4;5;6;12",
                      "END_F"]

    validate_space(space, make_element_invalid=_invalid)

    space = Permutations([4, 6, 6, 12, 5])
    assert space.min_value == 4
    assert space.max_value == 12
    assert space.dimension == 5
    assert str(space) == "permOfString"
    assert all(space.blueprint == np.array([4, 5, 6, 6, 12]))
    assert space.to_str(space.create()) == "4;5;6;6;12"
    assert not space.is_dense()
    assert space.n_points() == factorial(5) // 2

    validate_space(space, make_element_invalid=_invalid)

    with InMemoryLogger() as log:
        with log.key_values("F") as kv:
            space.log_parameters_to(kv)
        result = log.get_log()
    assert result == ["BEGIN_F",
                      "name: permOfString",
                      "class: moptipy.spaces.permutations.Permutations",
                      "nvars: 5",
                      "dtype: b",
                      "min: 4",
                      "max: 12",
                      "baseString: 4;5;6;6;12",
                      "END_F"]
