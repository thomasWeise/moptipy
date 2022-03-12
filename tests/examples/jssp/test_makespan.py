"""Test the makespan objective function of the JSSP."""
from moptipy.examples.jssp.makespan import Makespan
from moptipy.tests.on_jssp import validate_objective_on_jssp


def test_makespan_objective() -> None:
    """Test the makespan object."""
    validate_objective_on_jssp(Makespan)
