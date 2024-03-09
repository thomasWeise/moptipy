"""Test the callable objective function."""

from pycommons.io.temp import temp_file

from moptipy.api.component import Component
from moptipy.api.objective import Objective
from moptipy.tests.objective import validate_objective
from moptipy.utils.logger import FileLogger


class MyObjective(Objective):
    """The internal test objective."""

    def evaluate(self, x) -> int:
        """Evaluate a solution."""
        return 5

    def lower_bound(self) -> int:
        """Get the lower bound."""
        return -5

    def upper_bound(self) -> int:
        """Get the upper bound."""
        return 11

    def __str__(self):
        """Convert this object to a string."""
        return "hello"


def test_logged_args() -> None:
    """Test the logged arguments of the callable objective function."""
    f = MyObjective()
    assert isinstance(f, Objective)
    assert isinstance(f, Component)
    validate_objective(f, None, None)

    with temp_file() as path:
        with FileLogger(path) as log, log.key_values("F") as kv:
            f.log_parameters_to(kv)
        result = path.read_all_str().splitlines()

    assert result == [
        "BEGIN_F",
        "name: hello",
        "class: test_objective.MyObjective",
        "lowerBound: -5",
        "upperBound: 11",
        "END_F"]
