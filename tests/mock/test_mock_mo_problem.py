"""Test the mock multi-objective problem."""
from typing import Final

from numpy.random import Generator, default_rng

from moptipy.mock.mo_problem import MockMOProblem
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.mo_problem import validate_mo_problem
from moptipy.tests.on_bitstrings import random_bit_string
from moptipy.utils.nputils import DEFAULT_NUMERICAL, is_np_float, is_np_int


def test_mock_mo_problem() -> None:
    """Test the mock multi-objective problem."""
    random: Final[Generator] = default_rng()
    space: BitStrings = BitStrings(int(random.integers(2, 32)))

    for dtype in DEFAULT_NUMERICAL:
        n = int(random.integers(1, 10))
        mop = MockMOProblem.for_dtype(n, dtype)
        rdtype = mop.f_dtype()

        a = is_np_int(dtype)
        b = is_np_int(rdtype)
        if a != b:
            raise ValueError(
                f"is_int clash: should be {dtype}={a}, but is {rdtype}={b}")

        a = is_np_float(dtype)
        b = is_np_float(rdtype)
        if a != b:
            raise ValueError(
                f"is_float clash: should be {dtype}={a}, but is {rdtype}={b}")

        validate_mo_problem(
            mo_problem=mop,
            solution_space=space,
            make_solution_space_element_valid=random_bit_string,
            is_deterministic=True)
