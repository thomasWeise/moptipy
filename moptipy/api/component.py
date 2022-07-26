"""Provides the base class for all components of the moptipy API."""
from typing import Final

from moptipy.api import logging
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_name_of


class Component:
    """The base class for all components of the moptipy API."""

    def __repr__(self):
        """
        Get the string representation of this object.

        :return: the value returned by :meth:`__str__`
        """
        return str(self)

    def __str__(self):
        """
        Get the default to-string implementation returns the class name.

        :returns: the class name of this component

        >>> print(Component())
        Component
        """
        s: Final[str] = type_name_of(self)
        i: Final[int] = s.rfind(".")
        if i > 0:
            return s[i + 1:]
        return s

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        logger.key_value(logging.KEY_NAME, str(self))
        logger.key_value(logging.KEY_CLASS, type_name_of(self))
