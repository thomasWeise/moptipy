"""Test the callable objective function."""

# noinspection PyPackageRequirements

from moptipy.api.component import Component
from moptipy.api.objective import Objective
from moptipy.tests.objective import validate_objective
from moptipy.utils.logger import FileLogger
from moptipy.utils.temp import TempFile


class MyObjective(Objective):
    """The internal test objective."""
    def evaluate(self, x) -> int:
        return 5

    def lower_bound(self) -> int:
        return -5

    def upper_bound(self) -> int:
        return 11

    def __str__(self):
        return "hello"


def test_logged_args():
    """Test the logged arguments of the callable objective function."""
    f = MyObjective()
    assert isinstance(f, Objective)
    assert isinstance(f, Component)
    validate_objective(f, None, None)

    with TempFile.create() as path:
        with FileLogger(path) as log:
            with log.key_values("F") as kv:
                f.log_parameters_to(kv)
        result = path.read_all_list()

    assert result == [
        "BEGIN_F",
        "name: hello",
        "class: test_objective.MyObjective",
        "lowerBound: -5",
        "upperBound: 11",
        "END_F"]
