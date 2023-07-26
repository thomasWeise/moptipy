"""The base class with the information of a build."""

import codecs
import os.path
from io import TextIOBase
from re import MULTILINE
from re import compile as _compile
from typing import Final, Iterable, Pattern, cast

from moptipy.utils.strings import regex_sub
from moptipy.utils.types import type_error


def _canonicalize_path(path: str) -> str:
    """
    Check and canonicalize a path.

    :param path: the path
    :return: the canonicalized path
    """
    if not isinstance(path, str):
        raise type_error(path, "path", str)
    if len(path) <= 0:
        raise ValueError("Path must not be empty.")

    path = os.path.normcase(
        os.path.abspath(
            os.path.realpath(
                os.path.expanduser(
                    os.path.expandvars(path)))))
    if not isinstance(path, str):
        raise type_error(path, "canonicalized path", str)
    if len(path) <= 0:
        raise ValueError("Canonicalization must yield non-empty string, "
                         f"but returned {path!r}.")
    if path in [".", ".."]:
        raise ValueError(f"Canonicalization cannot yield {path!r}.")
    return path


#: the UTF-8 encoding
UTF8: Final[str] = "utf-8-sig"

#: The list of possible text encodings
__ENCODINGS: Final[tuple[tuple[tuple[bytes, ...], str], ...]] = \
    (((codecs.BOM_UTF8,), UTF8),
     ((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE), "utf-32"),
     ((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE), "utf-16"))


def _get_text_encoding(filename: str) -> str:
    """
    Get the text encoding from a BOM if present.

    Adapted from https://stackoverflow.com/questions/13590749.

    :param filename: the filename
    :return: the encoding
    """
    with open(filename, "rb") as f:
        header = f.read(4)  # Read just the first four bytes.
    for boms, encoding in __ENCODINGS:
        for bom in boms:
            if header.find(bom) == 0:
                return encoding
    return UTF8


#: a pattern used to clean up training white space
_PATTERN_TRAILING_WHITESPACE: Final[Pattern] = \
    _compile(r"[ \t]+\n", flags=MULTILINE)


class Path(str):
    """An immutable representation of a path."""

    def __new__(cls, value):  # noqa
        """
        Construct the object.

        :param value: the string value
        """
        return super().__new__(cls, _canonicalize_path(value))

    def enforce_file(self) -> None:
        """
        Enforce that the path references an existing file.

        :raises ValueError:  if `path` does not reference an existing file
        """
        if not self.is_file():
            raise ValueError(f"Path {self!r} does not identify a file.")

    def is_file(self) -> bool:
        """
        Check if this path identifies an existing file.

        :returns: True if this path identifies an existing file, False
            otherwise.
        """
        return os.path.isfile(self)

    def enforce_dir(self) -> None:
        """
        Enforce that the path references an existing directory.

        :raises ValueError:  if `path` does not reference an existing directory
        """
        if not os.path.isdir(self):
            raise ValueError(f"Path {self!r} does not identify a directory.")

    def contains(self, other: str) -> bool:
        """
        Check whether this path is a directory and contains another path.

        :param other: the other path
        :return: `True` is this path contains the other path, `False` of not
        """
        return os.path.isdir(self) and (
            os.path.commonpath([self])
            == os.path.commonpath([self, Path.path(other)]))

    def enforce_contains(self, other: str) -> None:
        """
        Raise an exception if this is not a directory containing another path.

        :param other: the other path
        :raises ValueError: if `other` is not a sub-path of this path
        """
        self.enforce_dir()
        if not self.contains(other):
            raise ValueError(f"Path {self!r} does not contain {other!r}.")

    def resolve_inside(self, relative_path: str) -> "Path":
        """
        Resolve a relative path to an absolute path inside this path.

        :param relative_path: the path to resolve
        :return: the resolved child path
        :raises ValueError: If the path would resolve to something outside of
            this path and/or if it is empty.
        """
        relative_path = relative_path.strip()
        if not relative_path:
            raise ValueError("Relative path cannot empty.")
        opath: Final[Path] = Path.path(os.path.join(self, relative_path))
        self.enforce_contains(opath)
        return opath

    def ensure_file_exists(self) -> bool:
        """
        Atomically ensure that the file exists and create it otherwise.

        :return: `True` if the file already existed and
            `False` if it was newly and atomically created.
        :raises: ValueError if anything goes wrong during the file creation
        """
        existed: bool = False
        try:
            os.close(os.open(self, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            existed = True
        except Exception as err:
            if isinstance(err, ValueError):
                raise
            raise ValueError(
                f"Error when trying to create file {self!r}.") from err
        self.enforce_file()
        return existed

    def ensure_dir_exists(self) -> None:
        """Make sure that the directory exists, create it otherwise."""
        try:
            os.makedirs(name=self, exist_ok=True)
        except FileExistsError:
            pass
        except Exception as err:
            if isinstance(err, ValueError):
                raise
            raise ValueError(
                f"Error when trying to create directory {self!r}.") from err
        self.enforce_dir()

    def open_for_read(self) -> TextIOBase:
        """
        Open this file for reading.

        :return: the file open for reading
        """
        return cast(TextIOBase, open(  # noqa
            self, encoding=_get_text_encoding(self),
            errors="strict"))

    def read_all_list(self) -> list[str]:
        """
        Read all the lines in a file.

        :return: the list of strings of text
        """
        self.enforce_file()
        with self.open_for_read() as reader:
            ret = reader.readlines()
        if not isinstance(ret, list):
            raise type_error(ret, f"return value of reading {self!r}", list)
        if len(ret) <= 0:
            raise ValueError(f"File {self!r} contains no text.")
        return [s.rstrip() for s in ret]

    def read_all_str(self) -> str:
        """
        Read a file as a single string.

        :return: the single string of text
        """
        self.enforce_file()
        with self.open_for_read() as reader:
            ret = reader.read()
        if not isinstance(ret, str):
            raise type_error(ret, f"return value of reading {self!r}", str)
        if len(ret) <= 0:
            raise ValueError(f"File {self!r} contains no text.")
        return ret

    def open_for_write(self) -> TextIOBase:
        """
        Open the file for writing.

        :return: the text io wrapper for writing
        """
        return cast(TextIOBase, open(  # noqa
            self, mode="w", encoding="utf-8", errors="strict"))

    def write_all(self, contents: str | Iterable[str]) -> None:
        """
        Write all the lines to this file.

        :param contents: the contents to write
        """
        self.ensure_file_exists()
        if not isinstance(contents, str | Iterable):
            raise type_error(contents, "contents", (str, Iterable))
        all_text = contents if isinstance(contents, str) \
            else "\n".join(contents)
        if len(all_text) <= 0:
            raise ValueError("Writing empty text is not permitted.")
        all_text = regex_sub(_PATTERN_TRAILING_WHITESPACE,
                             "\n", all_text.rstrip())
        if len(all_text) <= 0:
            raise ValueError(
                "Text becomes empty after removing trailing whitespace?")
        with self.open_for_write() as writer:
            writer.write(all_text)
            if all_text[-1] != "\n":
                writer.write("\n")

    def create_file_or_truncate(self) -> None:
        """
        Create the file identified by this path and truncate it if it exists.

        :raises: ValueError if anything goes wrong during the file creation
        """
        try:
            os.close(os.open(self, os.O_CREAT | os.O_TRUNC))
        except FileExistsError as err:
            raise ValueError(
                f"File {self!r} could not be truncated.") from err
        except Exception as err:
            raise ValueError(
                f"Error when trying to create  file {self!r}.") from err
        return self.enforce_file()

    @staticmethod
    def path(path: str) -> "Path":
        """
        Get a canonical path.

        :param path: the path to canonicalize
        :return: the `Path` instance
        """
        if isinstance(path, Path):
            return cast(Path, path)
        return Path(path)

    @staticmethod
    def file(path: str) -> "Path":
        """
        Get a path identifying a file.

        :param path: the path
        :return: the file
        """
        fi: Final[Path] = Path.path(path)
        fi.enforce_file()
        return fi

    @staticmethod
    def directory(path: str) -> "Path":
        """
        Get a path identifying a directory.

        :param path: the path
        :return: the file
        """
        fi: Final[Path] = Path.path(path)
        fi.enforce_dir()
        return fi
