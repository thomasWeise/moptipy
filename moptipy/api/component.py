"""Provides the base class for all components of the moptipy API."""
from typing import Callable, Final

from moptipy.api import logging
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import sanitize_name
from moptipy.utils.types import type_name_of, type_error


class Component:
    """The base class for all components of the moptipy API."""

    def __repr__(self):
        """
        Get the string representation of this object.

        :return: the value returned by :meth:`__str__`
        """
        return str(self)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        logger.key_value(logging.KEY_NAME, str(self))
        logger.key_value(logging.KEY_CLASS, type_name_of(self))


class CallableComponent(Component):
    """Wrap a Callable such as a lambda into a component."""

    def __init__(self,
                 inner: Callable,
                 name: str) -> None:
        """
        Create a wrapper mapping a Callable to an component.

        :param inner: the function to wrap, e.g., a lambda
        :param name: the name of the component
        :raises TypeError: if `inner` is not callable
        """
        if not callable(inner):
            raise type_error(inner, "inner function", call=True)

        #: the inner callable
        self._inner: Final[Callable] = inner
        #: the name of the component
        self.__name: Final[str] = sanitize_name(name)

    def __str__(self) -> str:
        """
        Get the name of this component.

        :return: the name
        """
        return self.__name

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_INNER_CLASS, type_name_of(self._inner))
