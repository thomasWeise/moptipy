"""
Code for testing of implementations of different parts of the moptipy API.

In this package, we provide code that can be used to test different parts of
the moptipy API and implementation. These are not unit tests, but rather code
that can be used to build unit tests. If you want to use moptipy in your code,
then likely you will implement own algorithms and operators. If you want to
test whether they comply with the moptipy specifications, then the functions
here will be helpful.
"""
from moptipy.tests.algorithm import test_algorithm, test_algorithm_on_jssp
from moptipy.tests.component import test_component
from moptipy.tests.encoding import test_encoding
from moptipy.tests.objective import test_objective
from moptipy.tests.operators import test_op0, test_op1
from moptipy.tests.space import test_space

__all__ = (
    "test_algorithm",
    "test_algorithm_on_jssp",
    "test_component",
    "test_encoding",
    "test_objective",
    "test_op0",
    "test_op1",
    "test_space")
