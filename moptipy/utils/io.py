"""A set of utilities interactions with the file system."""
import os
from os.path import realpath, isfile, isdir, abspath, expanduser, \
    expandvars, normcase
from shutil import rmtree
from tempfile import mkstemp, mkdtemp
from typing import Optional, Union, Tuple


def canonicalize_path(path: str) -> str:
    """
    An method which will check and canonicalize a path.

    :param str path: the path
    :return: the canonicalized path
    :rtype: str
    """
    if not isinstance(path, str):
        raise TypeError("path must be instance of str, but is "
                        + str(type(path)))
    if len(path) <= 0:
        raise ValueError("Path must not be empty.")

    path = normcase(abspath(realpath(expanduser(expandvars(path)))))
    if (not isinstance(path, str)) or (len(path) <= 0):
        raise ValueError(
            "Result of path canonicalization must be non-empty string, "
            "but is " + str(type(path)) + " with value '"
            + str(path) + "'.")
    return path


def enforce_file(path: str) -> str:
    """
    A method which enforces that a path references an existing file.

    :param path: the path identifying the file
    :return: the path
    :rtype: str
    :raises ValueError:  if `path` does not reference an existing file
    """
    if not isfile(path):
        raise ValueError("Path '" + path + "' does not identify file.")
    return path


def enforce_dir(path: str) -> str:
    """
    A method which enforces that a path references an existing directory.

    :param str path: the path identifying the directory
    :return: the path
    :rtype: str
    :raises ValueError:  if `path` does not reference an existing directory
    """
    if not isinstance(path, str):
        raise TypeError("path must be str, but is " + str(type(path)) + ".")
    if not isdir(path):
        raise ValueError("Path '" + path + "' does not identify file.")
    return path


class TempDir:
    """
    A scoped temporary directory to be used in a 'with' block.

    The directory and everything in it will be deleted upon exiting the
    'with' block. You can obtain its absolute/real path via :func:`str`.
    """

    def __init__(self, directory: Optional[str] = None) -> None:
        """
        Create the temporary directory.

        :param Optional[str] directory: an optional root directory
        :raises TypeError: if `directory` is not `None` but also no `str`
        """
        if not (directory is None):
            if not isinstance(directory, str):
                raise TypeError("directory must be str, but is "
                                + str(type(directory)) + ".")
            if len(directory) <= 0:
                raise ValueError("directory must not be empty string.")
        self.__path = enforce_dir(canonicalize_path(mkdtemp(dir=directory)))
        self.__open = True

    def __enter__(self) -> 'TempDir':
        """Nothing, just exists for `with`."""
        return self

    def close(self) -> None:
        """Delete the temporary directory and everything in it."""
        opn = self.__open
        self.__open = False
        if opn:
            rmtree(self.__path, ignore_errors=True, onerror=None)

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Call :meth:`close`.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        """
        self.close()

    def __str__(self) -> str:
        """
        Get the path to the temporary directory.

        :return: the path
        :rtype: str
        """
        return self.__path

    __repr__ = __str__


class TempFile:
    """
    A scoped temporary file to be used in a 'with' block.

    This file will be deleted upon exiting the 'with' block.
    You can obtain its absolute/real path via :func:`str`.
    """

    def __init__(self, directory: Union[TempDir, Optional[str]] = None,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None) -> None:
        """
        Create a temporary file.

        :param str directory: a root directory or `TempDir` instance
        :param Optional[str] prefix: an optional prefix
        :param Optional[str] suffix: an optional suffix, e.g., `.txt`
        :raises TypeError: if any of the parameters does not fulfill the type
            contract
        """
        if not (prefix is None):
            if not isinstance(prefix, str):
                raise TypeError("prefix must be str, but is "
                                + str(type(prefix)) + ".")
            if len(prefix) <= 0:
                raise ValueError("prefix must not be empty string.")

        if not (suffix is None):
            if not isinstance(suffix, str):
                raise TypeError("suffix must be str, but is "
                                + str(type(suffix)) + ".")
            if len(suffix) <= 0:
                raise ValueError("suffix must not be empty string.")

        usedir: Optional[str]
        if directory is None:
            usedir = None
        else:
            if isinstance(directory, TempDir):
                usedir = str(directory)
            else:
                usedir = directory
            if not isinstance(usedir, str):
                raise TypeError("directory must translate to str, but is "
                                + str(type(usedir)) + ".")
            if len(usedir) <= 0:
                raise ValueError("directory must not be empty string.")

        (handle, path) = mkstemp(suffix=suffix, prefix=prefix,
                                 dir=usedir)
        os.close(handle)
        self.__path = enforce_file(canonicalize_path(path))
        self.__open = True

    def __enter__(self) -> 'TempFile':
        """Nothing, just exists for `with`."""
        return self

    def close(self) -> None:
        """Delete the temporary file."""
        opn = self.__open
        self.__open = False
        if opn:
            os.remove(self.__path)

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Call :meth:`close`.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        """
        self.close()

    def __str__(self) -> str:
        """
        Get the path of this temp file.

        :return: the path of this temp file
        :rtype: str
        """
        return self.__path

    __repr__ = __str__


def file_create_or_fail(path: str) -> str:
    """
    Atomically create the file `path` and fail if it already exists.

    :param str path: the path
    :return: the canonicalized path
    :rtype: str
    :raises: ValueError if anything goes wrong during the file creation or
        if the file already exists
    """
    path = canonicalize_path(path)
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL))
    except FileExistsError as err:
        raise ValueError("File '" + path + "' already exists.") from err
    except Exception as err:
        raise ValueError("Error when trying to create  file'"
                         + path + "'.") from err
    return enforce_file(path)


def file_create_or_truncate(path: str) -> str:
    """
    Try to create the file identified by `path` and truncate it it it exists.

    :param str path: the path
    :return: the canonicalized path
    :rtype: str
    :raises: ValueError if anything goes wrong during the file creation
    """
    path = canonicalize_path(path)
    try:
        os.close(os.open(path, os.O_CREAT | os.O_TRUNC))
    except FileExistsError as err:
        raise ValueError("File '" + path + "' already exists.") from err
    except Exception as err:
        raise ValueError("Error when trying to create  file'"
                         + path + "'.") from err
    return enforce_file(path)


def file_ensure_exists(path: str) -> Tuple[str, bool]:
    """
    Atomically ensure that the file `path` exists and create it otherwise.

    :param str path: the path
    :return: a tuple `(path, existed)` with the canonicalized `path` and a
        Boolean value `existed` which is `True` if the file already existed and
        `False` if it was newly and atomically created.
    :rtype: Tuple[str, bool]
    :raises: ValueError if anything goes wrong during the file creation
    """
    path = canonicalize_path(path)
    existed = False
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        existed = True
    except Exception as err:
        raise ValueError("Error when trying to create  file'"
                         + path + "'.") from err
    return enforce_file(path), existed
