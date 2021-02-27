"""
In this package, we provide code that can be used to test different parts of
the moptipy API and implementation. These are not unit tests, but rather code
that can be used to build unit tests. If you want to use moptipy in your code,
then likely you will implement own algorithms and operators. If you want to
test whether they comply with the moptipy specifications, then the functions
here will be helpful.
"""
# noinspection PyUnresolvedReferences
from moptipy.version import __version__
from moptipy.tests.component import check_component
from moptipy.tests.objective import check_objective
from moptipy.tests.space import check_space

__all__ = ("check_component",
           "check_objective",
           "check_space")
