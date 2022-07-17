"""Test the basic archive pruner."""

from typing import List

import numpy as np

from moptipy.api.mo_archive import MOArchivePruner, MORecordY
from moptipy.tests.mo_archive_pruner import validate_mo_archive_pruner


def test_mo_archive_pruner() -> None:
    """Test the basic archive pruner."""
    pruner = MOArchivePruner()
    validate_mo_archive_pruner(pruner, range(1, 10))

    dt = np.dtype(int)
    orig: List[MORecordY] = [
        MORecordY(str(i), np.empty(2, dt), i) for i in range(10)]
    cpy = orig.copy()

    pruner.prune(orig, 8)
    assert orig[0].x == cpy[2].x
    assert orig[1].x == cpy[3].x
    assert orig[2].x == cpy[4].x
    assert orig[6].x == cpy[8].x
    assert orig[7].x == cpy[9].x
    assert orig[8].x == cpy[0].x
    assert orig[9].x == cpy[1].x
