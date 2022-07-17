"""Test the keep-farthest archive pruner."""

from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.mo.archive.keep_farthest import KeepFarthest
from moptipy.mo.problem.weighted_sum import WeightedSum
from moptipy.tests.mo_archive_pruner import validate_mo_archive_pruner


def test_keep_farthest_pruner() -> None:
    """Test the basic archive pruner."""
    for dim in range(1, 10):
        mop = WeightedSum([OneMax(12)] * dim, list(range(1, 1 + dim)))
        pruner = KeepFarthest(mop)
        validate_mo_archive_pruner(pruner, [dim])
