"""A set of utilities interactions with the file system."""
import os
from contextlib import AbstractContextManager
from shutil import rmtree
from tempfile import mkdtemp, mkstemp

from moptipy.utils.path import Path


class TempDir(Path, AbstractContextManager):
    """
    A scoped temporary directory to be used in a 'with' block.

    The directory and everything in it will be deleted upon exiting the
    'with' block.
    """

    #: is the directory open?
    __is_open: bool

    def __new__(cls, value):  # noqa
        """
        Construct the object.

        :param value: the string value
        """
        ret = super().__new__(cls, value)
        ret.enforce_dir()
        ret.__is_open = True
        return ret

    @staticmethod
    def create(directory: str | None = None) -> "TempDir":
        """
        Create the temporary directory.

        :param directory: an optional root directory
        :raises TypeError: if `directory` is not `None` but also no `str`
        """
        if directory is not None:
            root_dir = Path.path(directory)
            root_dir.enforce_dir()
        else:
            root_dir = None
        return TempDir(mkdtemp(dir=root_dir))

    def __enter__(self) -> "TempDir":
        """Nothing, just exists for `with`."""
        if not self.__is_open:
            raise ValueError(f"Temporary directory '{self}' already closed.")
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> bool:
        """
        Delete the temporary directory and everything in it.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        :returns: `True` to suppress an exception, `False` to rethrow it
        """
        opn = self.__is_open
        self.__is_open = False
        if opn:
            rmtree(self, ignore_errors=True, onerror=None)
        return exception_type is None


class TempFile(Path, AbstractContextManager):
    """
    A scoped temporary file to be used in a 'with' block.

    This file will be deleted upon exiting the 'with' block.
    """

    #: is the directory open?
    __is_open: bool

    def __new__(cls, value):  # noqa
        """
        Construct the object.

        :param value: the string value
        """
        ret = super().__new__(cls, value)
        ret.enforce_file()
        ret.__is_open = True
        return ret

    @staticmethod
    def create(directory: str | None = None,
               prefix: str | None = None,
               suffix: str | None = None) -> "TempFile":
        """
        Create a temporary file.

        :param directory: a root directory or `TempDir` instance
        :param prefix: an optional prefix
        :param suffix: an optional suffix, e.g., `.txt`
        :raises TypeError: if any of the parameters does not fulfill the type
            contract
        """
        if prefix is not None:
            prefix = prefix.strip()
            if not prefix:
                raise ValueError("Prefix cannot be empty if specified.")

        if suffix is not None:
            suffix = suffix.strip()
            if not suffix:
                raise ValueError("Prefix cannot be empty if specified.")

        if directory is not None:
            base_dir = Path.path(directory)
            base_dir.enforce_dir()
        else:
            base_dir = None

        (handle, path) = mkstemp(suffix=suffix, prefix=prefix, dir=base_dir)
        os.close(handle)
        return TempFile(path)

    def __enter__(self) -> "TempFile":
        """Nothing, just exists for `with`."""
        if not self.__is_open:
            raise ValueError(f"Temporary file '{self}' already deleted.")
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> bool:
        """
        Delete the temporary file.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        :returns: `True` to suppress an exception, `False` to rethrow it
        """
        opn = self.__is_open
        self.__is_open = False
        if opn:
            os.remove(self)
        return exception_type is None
