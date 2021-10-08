"""The base class with the information of a build."""

import codecs
import io
import os.path
from typing import cast, List, Iterable, Final, Union, Tuple


def _canonicalize_path(path: str) -> str:
    """
    An method which will check and canonicalize a path.

    :param str path: the path
    :return: the canonicalized path
    :rtype: str
    """
    if not isinstance(path, str):
        raise TypeError(
            f"path must be instance of str, but is {type(path)}.")
    if len(path) <= 0:
        raise ValueError("Path must not be empty.")

    path = os.path.normcase(
        os.path.abspath(
            os.path.realpath(
                os.path.expanduser(
                    os.path.expandvars(path)))))
    if not isinstance(path, str):
        raise TypeError("Path canonicalization should yield string, but "
                        f"returned {type(path)}.")
    if len(path) <= 0:
        raise ValueError("Canonicalization must yield non-empty string, "
                         f"but returned '{path}'.")
    if path in ['.', '..']:
        raise ValueError(f"Canonicalization cannot yield '{path}'.")
    return path


#: the UTF-8 encoding
UTF8: Final[str] = 'utf-8-sig'

#: The list of possible text encodings
__ENCODINGS: Final[Tuple[Tuple[Tuple[bytes, ...], str], ...]] = \
    (((codecs.BOM_UTF8,), UTF8),
     ((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE,), 'utf-32'),
     ((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE), 'utf-16'))


def _get_text_encoding(filename: str) -> str:
    """
    Get the text encoding from a BOM if present.

    Adapted from https://stackoverflow.com/questions/13590749.

    :param str filename: the filename
    :return: the encoding
    :rtype: str
    """
    with open(filename, 'rb') as f:
        header = f.read(4)  # Read just the first four bytes.
    for boms, encoding in __ENCODINGS:
        for bom in boms:
            if header.find(bom) == 0:
                return encoding
    return UTF8


class Path(str):
    """An immutable representation of a path."""

    def __new__(cls, value):
        """
        Construct the object.

        :param value: the string value
        """
        ret = super(Path, cls).__new__(cls, _canonicalize_path(value))
        return ret

    def enforce_file(self) -> None:
        """
        A method which enforces that a path references an existing file.

        :raises ValueError:  if `path` does not reference an existing file
        """
        if not os.path.isfile(self):
            raise ValueError(f"Path '{self}' does not identify a file.")

    def enforce_dir(self) -> None:
        """
        A method which enforces that a path references an existing directory.

        :raises ValueError:  if `path` does not reference an existing directory
        """
        if not os.path.isdir(self):
            raise ValueError(f"Path '{self}' does not identify a directory.")

    def contains(self, other: str) -> bool:
        """
        Check whether another path is contained in this path.

        :param str other: the other path
        :return: `True` is this path contains the other path, `False` of mpt
        :rtype: bool
        """
        return os.path.commonpath([self]) == \
            os.path.commonpath([self, Path.path(other)])

    def enforce_contains(self, other: str) -> None:
        """
        Raise an exception if this path does not contain the other path.

        :param str other: the other path
        :raises ValueError: if `other` is not a sub-path of this path
        """
        self.enforce_dir()
        if not self.contains(other):
            raise ValueError(f"Path '{self}' does not contain '{other}'.")

    def enforce_neither_contains(self, other: str) -> None:
        """
        Enforce that neither path contains another one.

        :param str other: the other path
        :raises ValueError: if `other` is contained in this path or vice versa
        """
        other_path: Final[Path] = Path.path(other)
        joint: Final[str] = os.path.commonpath([self, other_path])
        if joint == os.path.commonpath([self]):
            raise ValueError(f"Path '{self}' contains '{other_path}'.")
        if joint == os.path.commonpath([other_path]):
            raise ValueError(f"Path '{other_path}' contains '{self}'.")

    def relative_to(self, base_path: str) -> str:
        """
        Compute a relative path of this path towards the given base path.

        :param str base_path: the string
        :return: a relative path
        :rtype: str
        :raises ValueError: if this path is not inside `base_path`
        """
        opath: Final[Path] = Path.path(base_path)
        opath.enforce_contains(self)
        rpath = os.path.relpath(self, opath)
        if not rpath:
            raise ValueError(
                f"Relative path for '{self}' to '{opath}' is empty?")
        return rpath

    def resolve_inside(self, relative_path: str) -> 'Path':
        """
        Resolve a relative path to an absolute path inside this path.

        :param str relative_path: the path to resolve
        :return: the resolved child path
        :rtype: Path
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

        :return:  `True` if the file already existed and
            `False` if it was newly and atomically created.
        :rtype: bool
        :raises: ValueError if anything goes wrong during the file creation
        """
        existed: bool = False
        try:
            os.close(os.open(self, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            existed = True
        except Exception as err:
            raise ValueError(
                f"Error when trying to create file '{self}'.") from err
        self.enforce_file()
        return existed

    def ensure_dir_exists(self) -> None:
        """Make sure that the directory exists, create it otherwise."""
        os.makedirs(name=self, exist_ok=True)
        self.enforce_dir()

    def open_for_read(self) -> io.TextIOWrapper:
        """
        Open this file for reading.

        :return: the file open for reading
        :rtype: io.TextIOWrapper
        """
        return cast(io.TextIOWrapper, io.open(
            self, mode="rt", encoding=_get_text_encoding(self),
            errors="strict"))

    def read_all_list(self) -> List[str]:
        """
        Read all the lines in a file.

        :return: the list of strings of text
        :rtype: List[str]
        """
        self.enforce_file()
        with self.open_for_read() as reader:
            ret = reader.readlines()
        if not isinstance(ret, List):
            raise TypeError("List of strings expected, but "
                            f"found {type(ret)} in '{self}'.")
        if len(ret) <= 0:
            raise ValueError(f"File '{self}' contains no text.")
        return [s.rstrip() for s in ret]

    def read_all_str(self) -> str:
        """
        Read a file as a single string.

        :return: the single string of text
        :rtype: str
        """
        self.enforce_file()
        with self.open_for_read() as reader:
            ret = reader.read()
        if not isinstance(ret, str):
            raise TypeError("String expected, but "
                            f"found {type(ret)} in '{self}'.")
        if len(ret) <= 0:
            raise ValueError(f"File '{self}' contains no text.")
        return ret

    def open_for_write(self) -> io.TextIOWrapper:
        """
        Open the file for writing.

        :return: the text io wrapper for writing
        :rtype: io.TextIOWrapper
        """
        return cast(io.TextIOWrapper, io.open(
            self, mode="wt", encoding="utf-8", errors="strict"))

    def write_all(self, contents: Union[str, Iterable[str]]) -> None:
        """
        Write all the lines to this file.

        :param Iterable[str] contents: the contents to write
        """
        self.ensure_file_exists()
        if not isinstance(contents, (str, Iterable)):
            raise TypeError(
                f"Excepted str or Iterable, got {type(contents)}.")
        with self.open_for_write() as writer:
            all_text = contents if isinstance(contents, str) \
                else "\n".join(contents)
            if len(all_text) <= 0:
                raise ValueError("Writing empty text is not permitted.")
            writer.write(all_text)
            if all_text[-1] != "\n":
                writer.write("\n")

    def create_file_or_fail(self) -> None:
        """
        Atomically create the file path and fail if it already exists.

        :raises: ValueError if anything goes wrong during the file creation or
            if the file already exists
        """
        try:
            os.close(os.open(self, os.O_CREAT | os.O_EXCL))
        except FileExistsError as err:
            raise ValueError(f"File '{self}' already exists.") from err
        except Exception as err:
            raise ValueError(
                f"Error when trying to create  file '{self}'.") from err
        self.enforce_file()

    def create_file_or_truncate(self) -> None:
        """
        Create the file identified by this path and truncate it it it exists.

        :raises: ValueError if anything goes wrong during the file creation
        """
        try:
            os.close(os.open(self, os.O_CREAT | os.O_TRUNC))
        except FileExistsError as err:
            raise ValueError(f"File '{self}' already exists.") from err
        except Exception as err:
            raise ValueError(
                f"Error when trying to create  file '{self}'.") from err
        return self.enforce_file()

    @staticmethod
    def path(path: str) -> 'Path':
        """
        Get a canonical path.

        :param str path: the path to canonicalize
        :return: the `Path` instance
        :rtype: Path
        """
        if isinstance(path, Path):
            return cast(Path, path)
        return Path(path)

    @staticmethod
    def file(path: str) -> 'Path':
        """
        Get a path identifying a file.

        :param str path: the path
        :return: the file
        :rtype: Path
        """
        fi: Final[Path] = Path.path(path)
        fi.enforce_file()
        return fi

    @staticmethod
    def directory(path: str) -> 'Path':
        """
        Get a path identifying a directory.

        :param str path: the path
        :return: the file
        :rtype: Path
        """
        fi: Final[Path] = Path.path(path)
        fi.enforce_dir()
        return fi
