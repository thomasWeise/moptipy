from abc import ABC, abstractmethod
from moptipy.utils.logger import KeyValueSection
from moptipy.utils import logging
from typing import Callable


class Component(ABC):

    def __str__(self):
        return self.get_name()

    def __repr__(self):
        return self.get_name()

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    def log_parameters_to(self, logger: KeyValueSection):
        logger.key_value(logging.KEY_NAME, self.get_name())
        logger.key_value(logging.KEY_TYPE, str(type(self)))


class _CallableComponent(Component):
    """An internal base class for wrapping a Callable
    such as a lambda into an objective function."""

    def __init__(self,
                 inner: Callable,
                 name: str):
        """
        Create a wrapper mapping a Callable to an component

        :param Callable inner: the function to wrap, e.g., a lambda
        :param str name: the name of the component
        """

        if not isinstance(inner, Callable):
            raise ValueError("Inner function must be instance of Callable, "
                             "but is instance of " + str(type(inner)))

        self._inner = inner

        if name is None:
            ValueError("Name must not be None.")
        self.__name = logging.sanitize_name(name)

    def get_name(self) -> str:
        return self.__name

    def log_parameters_to(self, logger: KeyValueSection):
        super().log_parameters_to(logger)
        logger.key_value(logging.KEY_INNER_TYPE, str(type(self._inner)))
