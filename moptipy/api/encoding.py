"""
The base class for implementing encodings.

Sometimes, in optimization, the search space and the space of possible
solutions are different. For example, in the Job Shop Scheduling Problem
(JSSP), the search operators of the optimization algorithm may process
permutations with repetitions
(:class:`~moptipy.spaces.permutations.Permutations`) while the objective
function rates :class:`~moptipy.examples.jssp.gantt.Gantt` charts. The
Gantt charts are the solutions that the user wants, but they are harder to
deal with for search operators. However, we can easily develop search
operators for permutations and permutations can be mapped to Gantt charts
(see :class:`~moptipy.examples.jssp.ob_encoding.OperationBasedEncoding`).
A combination of search space, solution space, and encoding can be configured
when setting up an experiment :class:`~moptipy.api.execution.Execution`, which
then takes care of showing the search space to the optimization algorithm
while providing the objective function and user with the elements of the
solution spaces.

All encodings inherit from the class :class:`~moptipy.api.encoding.Encoding`.
If you implement a new such encoding, you may want to test it using the
pre-defined unit test routine
:func:`~moptipy.tests.encoding.validate_encoding`.
"""
from typing import Any

from moptipy.api.component import Component
from moptipy.utils.types import type_error


# start book
class Encoding(Component):
    """The encoding translates from a search space to a solution space."""

    def decode(self, x, y) -> None:
        """
        Translate from search- to solution space.

        Map a point `x` from the search space to a point `y`
        in the solution space.

        :param x: the point in the search space, remaining unchanged.
        :param y: the destination data structure for the point in the
            solution space, whose contents will be overwritten
        """
    # end book


def check_encoding(encoding: Any, none_is_ok: bool = True) -> Encoding | None:
    """
    Check whether an object is a valid instance of :class:`Encoding`.

    :param encoding: the object
    :param none_is_ok: is it ok if `None` is passed in?
    :return: the object
    :raises TypeError: if `encoding` is not an instance of :class:`Encoding`

    >>> check_encoding(Encoding())
    Encoding
    >>> check_encoding(Encoding(), True)
    Encoding
    >>> check_encoding(Encoding(), False)
    Encoding
    >>> try:
    ...     check_encoding('A')
    ... except TypeError as te:
    ...     print(te)
    encoding should be an instance of moptipy.api.encoding.\
Encoding but is str, namely 'A'.
    >>> try:
    ...     check_encoding('A', True)
    ... except TypeError as te:
    ...     print(te)
    encoding should be an instance of moptipy.api.encoding.\
Encoding but is str, namely 'A'.
    >>> try:
    ...     check_encoding('A', False)
    ... except TypeError as te:
    ...     print(te)
    encoding should be an instance of moptipy.api.encoding.\
Encoding but is str, namely 'A'.
    >>> print(check_encoding(None))
    None
    >>> print(check_encoding(None, True))
    None
    >>> try:
    ...     check_encoding(None, False)
    ... except TypeError as te:
    ...     print(te)
    encoding should be an instance of moptipy.api.encoding.\
Encoding but is None.
    """
    if isinstance(encoding, Encoding):
        return encoding
    if none_is_ok and (encoding is None):
        return None
    raise type_error(encoding, "encoding", Encoding)
