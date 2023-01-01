"""Test the sections parser."""

from moptipy.api import logging
from moptipy.evaluation.log_parser import LogParser
from moptipy.utils.logger import FileLogger
from moptipy.utils.temp import TempFile


class _TestParser(LogParser):
    def __init__(self, path: str):
        super().__init__()
        self.__state = 0
        self.__path = path

    def start_file(self, path: str) -> bool:
        assert self.__state == 0
        assert path == self.__path
        self.__state = 1
        return True

    def start_section(self, title: str) -> bool:
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

    def lines(self, lines: list[str]) -> bool:
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

    def end_file(self) -> bool:
        assert self.__state == 8
        assert isinstance(self.__path, str)
        self.__path = None
        self.__state = 0
        return True


def test_sections_parser() -> None:
    """Test parsing sections."""
    with TempFile.create(suffix=logging.FILE_SUFFIX) as tf:
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
        parser = _TestParser(str(tf))
        assert parser.parse_file(str(tf))
