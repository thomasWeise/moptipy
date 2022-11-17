"""A set of bounds, i.e., a minimal and a maximal value."""

from math import inf, isfinite
from typing import Final

from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import num_to_str_for_name
from moptipy.utils.types import type_error

#: the log key for the minimum value
KEY_MIN: Final[str] = "min"
#: the log key for the maximum value
KEY_MAX: Final[str] = "max"


class Bounds:
    """A set of bounds, i.e., a minimal and a maximal value."""

    def __init__(self, min_value, max_value) -> None:
        """
        Create the bounds object.

        :param min_value: the minimum permitted value
        :param max_value: the maximum permitted value
        """
        #: the lower bound, i.e., the minimum permitted value
        self.min_value: Final = min_value
        #: the upper bound, i.e., the maximum permitted value
        self.max_value: Final = max_value

    def __str__(self) -> str:
        """
        Get the name of these bounds.

        :return: the bounds separated by underscore

        >>> print(Bounds(-1, 1))
        m1_1
        >>> print(Bounds(0.5, 10.0))
        0d5_10
        >>> print(Bounds(-0.5, None))
        m0d5_n
        >>> print(Bounds(None, None))
        n_n
        """
        mi = "n" if self.min_value is None \
            else num_to_str_for_name(self.min_value)
        ma = "n" if self.max_value is None \
            else num_to_str_for_name(self.max_value)
        return f"{mi}_{ma}"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this bounds object to the given logger.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         Bounds(1, 2).log_parameters_to(kv)
        ...     print(l.get_log())
        ['BEGIN_C', 'min: 1', 'max: 2', 'END_C']
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         Bounds(1.0, None).log_parameters_to(kv)
        ...     print(l.get_log())
        ['BEGIN_C', 'min: 1', 'min(hex): 0x1.0000000000000p+0', 'END_C']
        """
        if self.min_value is not None:
            logger.key_value(KEY_MIN, self.min_value,
                             also_hex=isinstance(self.min_value, float))
        if self.max_value is not None:
            logger.key_value(KEY_MAX, self.max_value,
                             also_hex=isinstance(self.max_value, float))


class FloatBounds(Bounds):
    """A class representing floating point number bounds."""

    def __init__(self, min_value: float = -1.0,
                 max_value: float = 1.0) -> None:
        """
        Initialize the floating point bounds.

        :param min_value: the minimum value
        :param max_value: the maximum value
        """
        if not isinstance(min_value, float):
            raise type_error(min_value, "min_value", float)
        if not isfinite(min_value):
            raise ValueError(
                f"min_value must be finite, but is {min_value}.")
        if not isinstance(max_value, float):
            raise type_error(max_value, "max_value", float)
        if not isfinite(max_value):
            raise ValueError(
                f"max_value must be finite, but is {max_value}.")
        if min_value >= max_value:
            raise ValueError(
                f"max_value > min_value must hold, but got "
                f"min_value={min_value} and max_value={max_value}.")
        super().__init__(min_value, max_value)


class OptionalFloatBounds(Bounds):
    """A class representing optional floating point number bounds."""

    def __init__(self, min_value: float | None = None,
                 max_value: float | None = None) -> None:
        """
        Initialize the optional floating point bounds.

        :param min_value: the optional minimum value
        :param max_value: the optional maximum value
        """
        if min_value is not None:
            if not isinstance(min_value, float):
                raise type_error(min_value, "min_value", float)
            if not isfinite(min_value):
                if min_value <= -inf:
                    min_value = None
                else:
                    raise ValueError(
                        f"min_value must be finite, but is {min_value}.")
        if max_value is not None:
            if not isinstance(max_value, float):
                raise type_error(max_value, "max_value", float)
            if not isfinite(max_value):
                if max_value >= inf:
                    max_value = None
                else:
                    raise ValueError(
                        f"max_value must be finite, but is {max_value}.")
        if (min_value is not None) and (max_value is not None) and \
                (min_value >= max_value):
            raise ValueError(
                f"max_value > min_value must hold, but got "
                f"min_value={min_value} and max_value={max_value}.")
        super().__init__(min_value, max_value)


class IntBounds(Bounds):
    """A class representing integer bounds."""

    def __init__(self, min_value: int = 0, max_value: int = 1) -> None:
        """
        Initialize the integer bounds.

        :param min_value: the minimum value
        :param max_value: the maximum value
        """
        if not isinstance(min_value, int):
            raise type_error(min_value, "min_value", int)
        if not isinstance(max_value, int):
            raise type_error(max_value, "max_value", int)
        if min_value >= max_value:
            raise ValueError(
                f"max_value > min_value must hold, but got "
                f"min_value={min_value} and max_value={max_value}.")
        super().__init__(min_value, max_value)


class OptionalIntBounds(Bounds):
    """A class representing optional integer bounds."""

    def __init__(self, min_value: int | None = None,
                 max_value: int | None = None) -> None:
        """
        Initialize the optional integer bounds.

        :param min_value: the optional minimum value
        :param max_value: the optional maximum value
        """
        if min_value is not None:
            if not isinstance(min_value, int):
                raise type_error(min_value, "min_value", int)
        if max_value is not None:
            if not isinstance(max_value, int):
                raise type_error(max_value, "max_value", int)
            if (min_value is not None) and (min_value >= max_value):
                raise ValueError(
                    f"max_value > min_value must hold, but got "
                    f"min_value={min_value} and max_value={max_value}.")
        super().__init__(min_value, max_value)
