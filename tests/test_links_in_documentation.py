"""Test all the links in the project's *.md files."""
# noinspection PyPackageRequirements
from pycommons.dev.tests.links_in_md import check_links_in_md
from pycommons.io.path import file_path


def test_all_links_in_readme_md() -> None:
    """Test all the links in the README.md file."""
    check_links_in_md(file_path(__file__).up(2).resolve_inside("README.md"))


def test_all_links_in_contributing_md() -> None:
    """Test all the links in the CONTRIBUTING.md file."""
    check_links_in_md(file_path(__file__).up(2).resolve_inside(
        "CONTRIBUTING.md"))


def test_all_links_in_security_md() -> None:
    """Test all the links in the SECURITY.md file."""
    check_links_in_md(file_path(__file__).up(2).resolve_inside("SECURITY.md"))


def test_all_links_in_license() -> None:
    """Test all the links in the LICENSE file."""
    check_links_in_md(file_path(__file__).up(2).resolve_inside("LICENSE"))
