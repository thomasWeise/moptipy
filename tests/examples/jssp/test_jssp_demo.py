"""Test the demo functions."""
import numpy as np

from moptipy.examples.jssp.demo_examples import (
    demo_encoding,
    demo_gantt_chart,
    demo_instance,
    demo_point_in_search_space,
    demo_search_space,
    demo_solution,
    demo_solution_space,
)
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.spaces.permutations import Permutations
from moptipy.utils.temp import TempDir


def test_demo_instance() -> None:
    """Test the demo instance."""
    assert isinstance(demo_instance(), Instance)


def test_demo_search_space() -> None:
    """Test the demo search space."""
    assert isinstance(demo_search_space(), Permutations)


def test_demo_point_in_search_space() -> None:
    """Test the demo points in the search space."""
    assert isinstance(demo_point_in_search_space(True), np.ndarray)
    assert isinstance(demo_point_in_search_space(False), np.ndarray)


def test_demo_solution_space() -> None:
    """Test the demo solution space."""
    assert isinstance(demo_solution_space(), GanttSpace)


def test_demo_encoding() -> None:
    """Test the demo encoding."""
    assert isinstance(demo_encoding(), OperationBasedEncoding)


def test_demo_solution() -> None:
    """Test the demo solution."""
    for optimum in (False, True):
        s = demo_solution(optimum)
        assert isinstance(s, Gantt)
        s2 = demo_solution_space().create()
        p2 = demo_point_in_search_space(optimum)
        demo_encoding().decode(p2, s2)
        assert demo_solution_space().is_equal(s, s2)


def test_demo_chart() -> None:
    """Test the demo chart plotting."""
    for optimum in (False, True):
        for with_makespan in (False, True):
            for with_lower_bound in (False, True):
                with TempDir.create() as td:
                    lst = demo_gantt_chart(dirname=td,
                                           optimum=optimum,
                                           with_makespan=with_makespan,
                                           with_lower_bound=with_lower_bound)
                    assert isinstance(lst, list)
                    assert len(lst) >= 3
