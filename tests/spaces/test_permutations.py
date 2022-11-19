"""Test the space of permutations."""
import numpy as np

from moptipy.spaces.permutations import Permutations
from moptipy.tests.space import validate_space


def test_permutations_with_repetitions() -> None:
    """Test the permutations with repetitions."""
    def _invalid(x) -> np.ndarray:
        x[0] = 33
        return x

    validate_space(Permutations.with_repetitions(10, 1),
                   make_element_invalid=_invalid)
    validate_space(Permutations.with_repetitions(2, 1),
                   make_element_invalid=_invalid)
    validate_space(Permutations.with_repetitions(2, 12),
                   make_element_invalid=_invalid)
    validate_space(Permutations.with_repetitions(21, 11),
                   make_element_invalid=_invalid)


def test_permutations() -> None:
    """Test the permutation space."""
    def _invalid(x) -> np.ndarray:
        x[0] = 22
        return x

    validate_space(Permutations.standard(12), make_element_invalid=_invalid)
    validate_space(Permutations.standard(2), make_element_invalid=_invalid)
