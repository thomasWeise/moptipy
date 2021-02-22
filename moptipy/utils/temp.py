from tempfile import mkstemp, mkdtemp
from os import close, remove
from shutil import rmtree
from os.path import realpath
from typing import Optional, Union


class TempDir:
    """
    A scoped temporary directory to be used in a 'with' block
    and which will be deleted upon exiting the 'with' block.
    You can obtain its absolute/real path via str(..)
    """

    def __init__(self, directory: Optional[str] = None):
        """
        Create the temporary directory.
        :param str directory: an optional root directory
        """
        self.__path = realpath(mkdtemp(dir=directory))

    def __enter__(self):
        return self

    def close(self):
        """ Delete the temporary directory and everything in it."""
        if not (self.__path is None):
            rmtree(self.__path, ignore_errors=True, onerror=None)
            self.__path = None

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __str__(self):
        return self.__path

    __repr__ = __str__


class TempFile:
    """
    A scoped temporary file to be used in a 'with' block
    and which will be deleted upon exiting the 'with' block.
    You can obtain its absolute/real path via str(..)
    """

    def __init__(self, directory: Union[TempDir, Optional[str]] = None,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None):
        """
        Create a temporary file
        :param directory: a root directory or `TempDir` instance
        :param prefix: an optional prefix
        :param suffix: an optional suffix, e.g., `.txt`
        """
        (handle, path) = mkstemp(suffix=suffix, prefix=prefix, dir=None if (directory is None) else str(directory))
        close(handle)
        self.__path = realpath(path)

    def __enter__(self):
        return self

    def close(self):
        """Delete the temporary file."""
        if not (self.__path is None):
            remove(self.__path)
            self.__path = None

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __str__(self):
        return self.__path

    __repr__ = __str__
