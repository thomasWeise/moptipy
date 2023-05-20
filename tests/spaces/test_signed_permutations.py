"""Test the space of signed permutations."""
from math import factorial

import numpy as np

from moptipy.spaces.signed_permutations import SignedPermutations
from moptipy.tests.space import validate_space
from moptipy.utils.logger import InMemoryLogger


def test_signed_permutations_with_repetitions() -> None:
    """Test the signed permutations with repetitions."""
    def _invalid(x) -> np.ndarray:
        x[0] = 33
        return x

    space = SignedPermutations.with_repetitions(10, 1)
    assert space.min_value == -10
    assert space.max_value == 10
    assert space.unsigned_min_value == 1
    assert space.dimension == 10
    assert all(space.blueprint == np.array(list(range(1, 11))))
    assert space.n_points() == (2 ** 10) * factorial(10)
    assert space.to_str(space.create()) == "1;2;3;4;5;6;7;8;9;10"
    assert str(space) == "signedPerm10"
    assert space.is_dense()
    validate_space(space, make_element_invalid=_invalid)

    space = SignedPermutations.with_repetitions(2, 1)
    assert space.min_value == -2
    assert space.max_value == 2
    assert space.dimension == 2
    assert space.unsigned_min_value == 1
    assert str(space) == "signedPerm2"
    assert all(space.blueprint == np.array(list(range(1, 3))))
    assert space.n_points() == 8
    assert space.is_dense()
    assert space.to_str(space.create()) == "1;2"
    validate_space(space, make_element_invalid=_invalid)

    space = SignedPermutations.with_repetitions(2, 12)
    assert space.min_value == -2
    assert space.max_value == 2
    assert space.dimension == 2 * 12
    assert space.unsigned_min_value == 1
    assert str(space) == "signedPerm2w12r"
    assert all(space.blueprint == np.array(([1] * 12) + ([2] * 12)))
    assert space.n_points() == (2 ** 24) * (
        factorial(2 * 12) // factorial(12)) // factorial(12)
    assert space.is_dense()
    assert space.to_str(space.create()) == \
           "1;1;1;1;1;1;1;1;1;1;1;1;2;2;2;2;2;2;2;2;2;2;2;2"
    validate_space(space, make_element_invalid=_invalid)

    space = SignedPermutations.with_repetitions(21, 11)
    assert space.min_value == -21
    assert space.max_value == 21
    assert space.unsigned_min_value == 1
    assert space.dimension == 21 * 11
    assert str(space) == "signedPerm21w11r"
    assert all(space.blueprint == np.array(sorted(list(range(1, 22)) * 11)))
    assert space.is_dense()
    validate_space(space, make_element_invalid=_invalid)


def test_standard_signed_permutations() -> None:
    """Test the signed permutation space."""
    def _invalid(x) -> np.ndarray:
        x[0] = 111
        return x

    space = SignedPermutations.standard(12)
    assert space.min_value == -12
    assert space.max_value == 12
    assert space.dimension == 12
    assert space.unsigned_min_value == 1
    assert all(space.blueprint == np.array(list(range(1, 13))))
    assert space.n_points() == (2 ** 12) * factorial(12)
    assert str(space) == "signedPerm12"
    assert space.is_dense()

    validate_space(space, make_element_invalid=_invalid)

    space = SignedPermutations.standard(2)
    assert space.min_value == -2
    assert space.max_value == 2
    assert space.unsigned_min_value == 1
    assert space.dimension == 2
    assert space.n_points() == 8
    assert all(space.blueprint == np.array(list(range(1, 3))))
    assert str(space) == "signedPerm2"
    assert space.is_dense()

    validate_space(space, make_element_invalid=_invalid)


def test_permutations_blueprint() -> None:
    """Test the signed permutation space."""
    def _invalid(x) -> np.ndarray:
        x[0] = 111
        return x

    space = SignedPermutations([4, 6, 12, 5])
    assert space.min_value == -12
    assert space.unsigned_min_value == 4
    assert space.max_value == 12
    assert space.dimension == 4
    assert str(space) == "signedPermOfString"
    assert all(space.blueprint == np.array([4, 5, 6, 12]))
    assert not space.is_dense()
    assert space.n_points() == (2 ** 4) * factorial(4)
    assert space.to_str(space.create()) == "4;5;6;12"

    with InMemoryLogger() as log:
        with log.key_values("F") as kv:
            space.log_parameters_to(kv)
        result = log.get_log()
    assert result == [
        "BEGIN_F",
        "name: signedPermOfString",
        "class: moptipy.spaces.signed_permutations.SignedPermutations",
        "nvars: 4",
        "dtype: b",
        "min: -12",
        "max: 12",
        "unsignedMin: 4",
        "repetitions: 1",
        "baseString: 4;5;6;12",
        "END_F"]

    validate_space(space, make_element_invalid=_invalid)

    space = SignedPermutations([4, 6, 6, 12, 5])
    assert space.min_value == -12
    assert space.unsigned_min_value == 4
    assert space.max_value == 12
    assert space.dimension == 5
    assert str(space) == "signedPermOfString"
    assert all(space.blueprint == np.array([4, 5, 6, 6, 12]))
    assert space.to_str(space.create()) == "4;5;6;6;12"
    assert not space.is_dense()
    assert space.n_points() == (2 ** 5) * (factorial(5) // 2)

    validate_space(space, make_element_invalid=_invalid)

    with InMemoryLogger() as log:
        with log.key_values("F") as kv:
            space.log_parameters_to(kv)
        result = log.get_log()
    assert result == [
        "BEGIN_F",
        "name: signedPermOfString",
        "class: moptipy.spaces.signed_permutations.SignedPermutations",
        "nvars: 5",
        "dtype: b",
        "min: -12",
        "max: 12",
        "unsignedMin: 4",
        "baseString: 4;5;6;6;12",
        "END_F"]
