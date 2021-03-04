"""
In this package, we provide code that can be used to test different parts of
the moptipy API and implementation. These are not unit tests, but rather code
that can be used to build unit tests. If you want to use moptipy in your code,
then likely you will implement own algorithms and operators. If you want to
test whether they comply with the moptipy specifications, then the functions
here will be helpful.
"""
from moptipy.tests.algorithm import check_algorithm
from moptipy.tests.component import check_component
from moptipy.tests.encoding import check_encoding
from moptipy.tests.objective import check_objective
from moptipy.tests.operators import check_op0, check_op1
from moptipy.tests.space import check_space
# noinspection PyUnresolvedReferences
from moptipy.version import __version__

__all__ = ("check_algorithm",
           "check_component",
           "check_encoding",
           "check_objective",
           "check_op0",
           "check_op1",
           "check_space")
