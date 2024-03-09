"""Test all the example code in the project's README.md file."""

from pycommons.dev.tests.examples_in_md import check_examples_in_md
from pycommons.io.path import file_path


def test_all_examples_from_readme_md() -> None:
    """Test all the example Python codes in the README.md file."""
    # First, we load the README.md file as a single string
    check_examples_in_md(file_path(__file__).up(2).resolve_inside("README.md"))
