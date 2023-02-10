"""A base class for bitstring-based problems."""

from typing import Final

from moptipy.api.objective import Objective
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import check_int_range


class BitStringProblem(Objective):
    """
    A base class for problems defined over bit strings.

    This base class has a set of default behaviors. It has an attribute
    :attr:`n` denoting the lengths of the accepted bit strings. Its
    :meth:`lower_bound` returns `0` and its :meth:`upper_bound` returns
    :attr:`n`. :meth:`is_always_integer` returns `True`.
    """

    def __init__(self, n: int) -> None:  # +book
        """
        Initialize the bitstring objective function.

        :param n: the dimension of the problem
        """
        super().__init__()
        #: the length of the bit strings
        self.n: Final[int] = check_int_range(n, "n", 1)

    def lower_bound(self) -> int:
        """
        Get the lower bound of the bit string based problem.

        :return: 0

        >>> print(BitStringProblem(10).lower_bound())
        0
        """
        return 0

    def upper_bound(self) -> int:
        """
        Get the upper bound of the bit string based problem.

        :return: the length of the bit string

        >>> print(BitStringProblem(7).upper_bound())
        7
        """
        return self.n

    def is_always_integer(self) -> bool:
        """
        Return `True` if the `evaluate` function always returns an `int`.

        This pre-defined function for bit-string based problems will always
        return `True`.

        :retval True: always
        """
        return True

    def space(self) -> BitStrings:
        """
        Create a proper search space for this problem.

        :returns: an instance of
            :class:`~moptipy.spaces.bitstrings.BitStrings` for bit strings of
            length :attr:`n`
        """
        return BitStrings(self.n)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         BitStringProblem(5).log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[1]
        'name: BitStringProblem'
        >>> text[3]
        'lowerBound: 0'
        >>> text[4]
        'upperBound: 5'
        >>> text[5]
        'n: 5'
        >>> len(text)
        7
        """
        super().log_parameters_to(logger)
        logger.key_value("n", self.n)
