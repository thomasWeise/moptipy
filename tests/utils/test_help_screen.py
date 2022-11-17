"""Test the help screen."""

import os.path

from moptipy.utils.help import help_screen


def test_help_screen() -> None:
    """Test the call to the help screen method."""
    path = os.path.abspath("../../moptipy/utils/cache.py")
    help_screen("Test", path)
    help_screen("Test", path, "bla-text", [
        ("a", "bla"), ("b", "bbb", False), ("c", "bbb", True)])
