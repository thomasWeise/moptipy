"""Test the mock objective function."""
from typing import Final

from numpy.random import Generator, default_rng

from moptipy.mock.objective import MockObjective
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.objective import validate_objective
from moptipy.tests.on_bitstrings import random_bit_string
from moptipy.utils.nputils import DEFAULT_NUMERICAL


def test_mock_int_objective() -> None:
    """Test the mock integer objective function."""
    random: Final[Generator] = default_rng()
    space: BitStrings = BitStrings(int(random.integers(2, 32)))

    lb = int(random.integers(0, 100))
    ub = lb + int(random.integers(1000, 100000))

    f: MockObjective = MockObjective(is_int=True, lb=lb, ub=ub)

    validate_objective(
        objective=f,
        solution_space=space,
        make_solution_space_element_valid=random_bit_string,
        is_deterministic=True,
        lower_bound_threshold=lb,
        upper_bound_threshold=ub)


def test_mock_float_objective() -> None:
    """Test the mock float objective function."""
    random: Final[Generator] = default_rng()
    space: BitStrings = BitStrings(int(random.integers(2, 32)))

    lb = float(random.uniform(-100, 100))
    ub = lb + int(random.uniform(1000, 100000))

    f: MockObjective = MockObjective(is_int=False, lb=lb, ub=ub)

    validate_objective(
        objective=f,
        solution_space=space,
        make_solution_space_element_valid=random_bit_string,
        is_deterministic=True,
        lower_bound_threshold=lb,
        upper_bound_threshold=ub)


def test_mock_auto_objective() -> None:
    """Test the mock float objective function."""
    random: Final[Generator] = default_rng()
    space: BitStrings = BitStrings(int(random.integers(2, 32)))

    for dtype in DEFAULT_NUMERICAL:
        validate_objective(
            objective=MockObjective.for_type(dtype),
            solution_space=space,
            make_solution_space_element_valid=random_bit_string,
            is_deterministic=True)
