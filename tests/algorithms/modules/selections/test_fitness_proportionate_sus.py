"""Test the stochastic uniform sampling selection algorithm."""

from moptipy.algorithms.modules.selections.fitness_proportionate_sus import (
    FitnessProportionateSUS,
)
from moptipy.tests.selection import validate_selection


def test_sus_0() -> None:
    """Test the stochastic uniform sampling selection strategy."""
    validate_selection(FitnessProportionateSUS(), False)


def test_sus_0_01() -> None:
    """Test the stochastic uniform sampling selection strategy."""
    validate_selection(FitnessProportionateSUS(0.01), False,
                       upper_source_size_limit=99)


def test_sus_0_1() -> None:
    """Test the stochastic uniform sampling selection strategy."""
    validate_selection(FitnessProportionateSUS(0.1), False,
                       upper_source_size_limit=9)


def test_sus_0_025() -> None:
    """Test the stochastic uniform sampling selection strategy."""
    validate_selection(FitnessProportionateSUS(0.025), False,
                       upper_source_size_limit=int(1.0 / 0.025) - 1)
