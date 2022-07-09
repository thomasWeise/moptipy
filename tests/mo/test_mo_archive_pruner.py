"""Test the basic archive pruner."""

from typing import Final, List, Tuple

import numpy as np

from moptipy.api.mo_archive_pruner import MOArchivePruner
from moptipy.tests.mo_archive_pruner import validate_mo_archive_pruner


def test_mo_archive_pruner() -> None:
    """Test the basic archive pruner."""
    pruner = MOArchivePruner()
    validate_mo_archive_pruner(pruner, range(1, 10))

    dt = np.dtype(int)
    orig: Final[List[Tuple[str, np.ndarray]]] = [
        (str(i), np.empty(2, dt)) for i in range(10)]
    cpy = orig.copy()

    pruner.prune(orig, 8)
    assert orig[0][0] == cpy[2][0]
    assert orig[1][0] == cpy[3][0]
    assert orig[2][0] == cpy[4][0]
    assert orig[6][0] == cpy[8][0]
    assert orig[7][0] == cpy[9][0]
    assert orig[8][0] == cpy[0][0]
    assert orig[9][0] == cpy[1][0]
