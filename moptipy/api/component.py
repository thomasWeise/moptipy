"""
This module provides :class:`Component`, the base class for all components
of the moptipy API.
"""
from abc import ABC, abstractmethod
from typing import Callable

from moptipy.utils import logging
from moptipy.utils.logger import KeyValueSection


class Component(ABC):
    """The base class for all components of the moptipy API."""

    def __str__(self):
        return self.get_name()

    def __repr__(self):
        return self.get_name()

    @abstractmethod
    def get_name(self) -> str:
        """
        Get a canonical name of this component.
        :return: the canonical name of this component
        :rtype: str
        """
        raise NotImplementedError

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log all parameters of this component as key-value pairs to the given
        logger.
        :param KeyValueSection logger:
        """
        logger.key_value(logging.KEY_NAME, self.get_name())
        logger.key_value(logging.KEY_TYPE, str(type(self)))


class _CallableComponent(Component):
    """An internal base class for wrapping a Callable
    such as a lambda into an objective function."""

    def __init__(self,
                 inner: Callable,
                 name: str) -> None:
        """
        Create a wrapper mapping a Callable to an component

        :param Callable inner: the function to wrap, e.g., a lambda
        :param str name: the name of the component
        :raises TypeError: if `inner` is not callable
        :raises ValueError: if name is `None`
        """

        if not callable(inner):
            raise TypeError("Inner function must be callable, but is a "
                            + str(type(inner)))

        self._inner = inner
        self.__name = logging.sanitize_name(name)

    def get_name(self) -> str:
        return self.__name

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_INNER_TYPE, str(type(self._inner)))
