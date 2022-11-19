"""Test the frequency fitness assignment strategy."""

from typing import Final

from numpy.random import Generator, default_rng

from moptipy.algorithms.so.fitnesses.ffa import FFA
from moptipy.api.objective import Objective
from moptipy.tests.on_bitstrings import validate_fitness_on_bitstrings


def test_ffa_on_bit_strings() -> None:
    """Test the frequency fitness assignment process on bit strings."""
    validate_fitness_on_bitstrings(
        fitness=lambda f: FFA(f),
        class_needed="moptipy.algorithms.so.fitnesses.ffa._IntFFA1")


def test_ffa_on_bit_strings_2() -> None:
    """Test the frequency fitness assignment process on bit strings."""

    def prepare(f: Objective) -> Objective:
        random: Final[Generator] = default_rng()
        olb = f.lower_bound()
        oub = f.upper_bound()
        while True:
            lb = int(random.integers(-100_000_000, 100_000_000))
            ub = int(random.integers(lb + oub - olb + 1, lb + 100_000_000))
            if ((ub - lb) <= 100_000_000) and (not (0 <= lb <= 10_000_000)):
                break
        f.lower_bound = lambda _l=lb: _l  # type: ignore
        f.upper_bound = lambda _u=ub: _u  # type: ignore
        ofe = f.evaluate
        f.evaluate = lambda x, _l=lb - olb, _o=ofe: _o(x) + _l  # type: ignore
        return f

    validate_fitness_on_bitstrings(
        fitness=lambda f: FFA(f),
        class_needed="moptipy.algorithms.so.fitnesses.ffa._IntFFA2",
        prepare_objective=prepare)


def test_ffa_on_bit_strings_3() -> None:
    """Test the dict-based frequency fitness assignment on bit strings."""

    def prepare(f: Objective) -> Objective:
        random: Final[Generator] = default_rng()
        lb = olb = f.lower_bound()
        oub = f.upper_bound()
        choice: Final[int] = random.integers(1, 4)

        if choice in (1, 3):
            while True:
                lb = int(random.integers(-1_000_000_000, 1_000_000_000))
                ub = int(random.integers(lb + oub - olb + 1,
                                         lb + 10_000_000_000))
                if (ub - lb) > 100_000_000:
                    break
            f.lower_bound = lambda _l=lb: _l  # type: ignore
            f.upper_bound = lambda _u=ub: _u  # type: ignore
            ofe = f.evaluate
            f.evaluate = \
                lambda x, _l=lb - olb, _o=ofe: _o(x) + _l  # type: ignore

        if choice in (2, 3):
            f.is_always_integer = lambda: False
            ofe2 = f.evaluate
            f.evaluate = lambda x, _of=ofe2: float(_of(x))  # type: ignore

        return f

    validate_fitness_on_bitstrings(
        fitness=lambda f: FFA(f),
        class_needed="moptipy.algorithms.so.fitnesses.ffa._DictFFA",
        prepare_objective=prepare)
