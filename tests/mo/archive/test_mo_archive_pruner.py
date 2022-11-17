"""Test the basic archive pruner."""


import numpy as np

from moptipy.api.mo_archive import MOArchivePruner, MORecord
from moptipy.tests.mo_archive_pruner import validate_mo_archive_pruner


def test_mo_archive_pruner() -> None:
    """Test the basic archive pruner."""
    pruner = MOArchivePruner()
    validate_mo_archive_pruner(lambda _: pruner, range(1, 10))

    dt = np.dtype(int)
    orig: list[MORecord] = [
        MORecord(str(i), np.empty(2, dt)) for i in range(10)]
    cpy = orig.copy()

    pruner.prune(orig, 8, len(orig))
    assert orig[0].x == cpy[2].x
    assert orig[1].x == cpy[3].x
    assert orig[2].x == cpy[4].x
    assert orig[6].x == cpy[8].x
    assert orig[7].x == cpy[9].x
    assert orig[8].x == cpy[0].x
    assert orig[9].x == cpy[1].x
