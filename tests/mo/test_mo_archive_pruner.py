"""Test the basic archive pruner."""

from typing import List, Tuple, Any

import numpy as np

from moptipy.api.mo_archive_pruner import MOArchivePruner
from moptipy.tests.mo_archive_pruner import validate_mo_archive_pruner


def test_mo_archive_pruner() -> None:
    """Test the basic archive pruner."""
    pruner = MOArchivePruner()
    validate_mo_archive_pruner(pruner, range(1, 10))

    dt = np.dtype(int)
    orig: List[Tuple[np.ndarray, Any]] = [
        (np.empty(2, dt), str(i)) for i in range(10)]
    cpy = orig.copy()

    pruner.prune(orig, 8)
    assert orig[0][1] == cpy[2][1]
    assert orig[1][1] == cpy[3][1]
    assert orig[2][1] == cpy[4][1]
    assert orig[6][1] == cpy[8][1]
    assert orig[7][1] == cpy[9][1]
    assert orig[8][1] == cpy[0][1]
    assert orig[9][1] == cpy[1][1]
