"""A class for representing Gantt charts as objects."""
from typing import Final

import numpy as np

from moptipy.api.logging import SECTION_RESULT_Y, SECTION_SETUP
from moptipy.evaluation.log_parser import LogParser
from moptipy.examples.jssp.instance import Instance
from moptipy.utils.types import type_error


# start book
class Gantt(np.ndarray):
    """
    A class representing Gantt charts.

    A Gantt chart is a diagram that visualizes when a job on a given
    machine begins or ends. We here represent it as a three-dimensional
    matrix. This matrix has one row for each machine and one column for
    each operation on the machine.
    In each cell, it holds three values: the job ID, the start, and the
    end time of the job on the machine. The Gantt chart has the
    additional attribute `instance` which references the JSSP instance
    for which the chart is constructed.
    Gantt charts must only be created by an instance of
    :class:`moptipy.examples.jssp.gantt_space.GanttSpace`.
    """

# end book

    #: the JSSP instance for which the Gantt chart is created
    instance: Instance

    def __new__(cls, space) -> "Gantt":
        """
        Create the Gantt chart.

        :param space: the Gantt space for which the instance is created.
        """
        gnt: Final[Gantt] = np.ndarray.__new__(Gantt, space.shape, space.dtype)
        #: the JSSP instance for which the Gantt chart is created
        gnt.instance = space.instance
        return gnt

    @staticmethod
    def from_log(file: str,
                 instance: Instance | None = None) -> "Gantt":
        """
        Load a Gantt chart from a log file.

        :param file: the log file path
        :param instance: the optional JSSP instance: if `None` is provided,
            we try to load it from the resources
        :returns: the Gantt chart
        """
        parser: Final[_GanttParser] = _GanttParser(instance)
        parser.parse_file(file)
        # noinspection PyProtectedMember
        res = parser._result
        if res is None:
            raise ValueError("Failed to load Gantt chart.")
        return res


class _GanttParser(LogParser):
    """The log parser for loading Gantt charts."""

    def __init__(self, instance: Instance | None = None):
        """
        Create the gantt parser.

        :param instance: the optional JSSP instance: if `None` is provided,
            we try to load it from the resources
        """
        super().__init__()
        if (instance is not None) and (not isinstance(instance, Instance)):
            raise type_error(instance, "instance", Instance)
        #: the internal instance
        self.__instance: Instance | None = instance
        #: the internal section mode: 0=none, 1=setup, 2=y
        self.__sec_mode: int = 0
        #: the gantt string
        self.__gantt_str: str | None = None
        #: the result Gantt chart
        self._result: Gantt | None = None

    def start_section(self, title: str) -> bool:
        """Start a section."""
        super().start_section(title)
        self.__sec_mode = 0
        if title == SECTION_SETUP:
            if self.__instance is None:
                self.__sec_mode = 1
                return True
            return False
        if title == SECTION_RESULT_Y:
            self.__sec_mode = 2
            return True
        return False

    def lines(self, lines: list[str]) -> bool:
        """Parse the lines."""
        if self.__sec_mode == 1:
            if self.__instance is not None:
                raise ValueError(
                    f"instance is already set to {self.__instance}.")
            key: Final[str] = "y.inst.name: "
            for line in lines:
                if line.startswith(key):
                    self.__instance = Instance.from_resource(
                        line[len(key):].strip())
            if self.__instance is None:
                raise ValueError(f"Did not find instance key {key!r} "
                                 f"in section {SECTION_SETUP}!")
        elif self.__sec_mode == 2:
            self.__gantt_str = " ".join(lines).strip()
        else:
            raise ValueError("Should not be in section?")
        return (self.__instance is None) or (self.__gantt_str is None)

    def end_file(self) -> bool:
        """End the file."""
        if self.__gantt_str is None:
            raise ValueError(f"Section {SECTION_RESULT_Y} missing!")
        if self.__instance is None:
            raise ValueError(f"Section {SECTION_SETUP} missing or empty!")
        if self._result is not None:
            raise ValueError("Applied parser to more than one log file?")
        # pylint: disable=C0415,R0401
        from moptipy.examples.jssp.gantt_space import (
            GanttSpace,  # pylint: disable=C0415,R0401
        )

        self._result = GanttSpace(self.__instance).from_str(self.__gantt_str)
        self.__gantt_str = None
        return False
