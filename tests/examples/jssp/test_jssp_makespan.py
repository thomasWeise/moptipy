"""Test the makespan objective function of the JSSP."""
from moptipy.examples.jssp.demo_examples import demo_instance, demo_solution
from moptipy.examples.jssp.makespan import Makespan
from moptipy.tests.on_jssp import validate_objective_on_jssp


def test_makespan_objective() -> None:
    """Test the makespan objective function."""
    validate_objective_on_jssp(Makespan)

    demo = demo_instance()
    sol = demo_solution(True)
    ms = Makespan(demo)
    assert ms.evaluate(sol) == 180
    assert ms.lower_bound() == 180
    assert ms.is_always_integer()
