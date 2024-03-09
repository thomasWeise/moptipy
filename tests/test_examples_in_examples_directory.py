"""Test all the example code in the project's examples directory."""


from pycommons.dev.tests.examples_in_dir import check_examples_in_dir
from pycommons.io.path import file_path


def test_examples_in_examples_directory() -> None:
    """Test all the examples in the examples directory."""
    # First, we resolve the directories
    check_examples_in_dir(file_path(__file__).up(2).resolve_inside(
        "examples"))
