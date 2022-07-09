"""Test the keep-farthest archive pruner."""

from moptipy.mo.keep_farthest import KeepFarthest
from moptipy.tests.mo_archive_pruner import validate_mo_archive_pruner


def test_keep_farthest_pruner() -> None:
    """Test the basic archive pruner."""
    for dim in range(1, 10):
        pruner = KeepFarthest(dim)
        validate_mo_archive_pruner(pruner, [dim])
