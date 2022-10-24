"""Test the worktime objective function of the JSSP."""
from moptipy.examples.jssp.worktime import Worktime
from moptipy.tests.on_jssp import validate_objective_on_jssp


def test_worktime_objective() -> None:
    """Test the worktime objective function."""
    validate_objective_on_jssp(Worktime)
