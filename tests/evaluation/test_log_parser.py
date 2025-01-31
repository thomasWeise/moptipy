"""Test the sections parser."""

from pycommons.io.path import Path
from pycommons.io.temp import temp_file

from moptipy.api import logging
from moptipy.evaluation.log_parser import LogParser
from moptipy.utils.logger import FileLogger


class _TestParser(LogParser[bool]):
    def __init__(self, path: Path):
        super().__init__()
        self.__state = 0
        self.__path = path

    def _should_parse_file(self, file: Path) -> bool:
        assert self.__state == 0
        assert file == self.__path
        self.__state = 1
        return True

    def _parse_file(self, file: Path) -> bool:
        super()._parse_file(file)
        return True

    def _start_section(self, title: str) -> bool:
        if self.__state == 1:
            assert title == "TEXT"
            self.__state = 2
            return True
        if self.__state == 3:
            assert title == "CSV"
            self.__state = 4
            return True
        if self.__state == 5:
            assert title == "SKIP1"
            self.__state = 6
            return False
        if self.__state == 6:
            assert title == "KV"
            self.__state = 7
            return True
        raise AssertionError("Should never get here.")

    def _lines(self, lines: list[str]) -> bool:
        if self.__state == 2:
            assert lines == ["a", "b", "c"]
            self.__state = 3
            return True
        if self.__state == 4:
            assert len(lines) == 101
            self.__state = 5
            return True
        if self.__state == 7:
            assert lines == ["k: l", "m: n"]
            self.__state = 8
            return False
        raise AssertionError("Should never get here.")

    def _end_parse_file(self, file: Path) -> None:
        assert self.__state == 8
        assert isinstance(self.__path, str)
        assert file == self.__path
        self.__path = None
        self.__state = 0


def test_sections_parser() -> None:
    """Test parsing sections."""
    with temp_file(suffix=logging.FILE_SUFFIX) as tf:
        with FileLogger(tf) as logger:
            with logger.text("TEXT") as txt:
                txt.write("a\nb\nc")
            with logger.csv("CSV", ["d", "e", "f"]) as csv:
                for i in range(100):
                    csv.row([1 + 3 * i, 2 + 3 * i, 3 + 3 * i])
            with logger.key_values("SKIP1") as skip:
                skip.key_value("h", "i")
            with logger.key_values("KV") as kv:
                kv.key_value("k", "l")
                kv.key_value("m", "n")
            with logger.key_values("SKIP2") as skip:
                skip.key_value("x", "y")
        parser = _TestParser(tf)
        assert all(parser.parse(str(tf)))
