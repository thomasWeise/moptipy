"""A base class for logging improvements."""
from os import remove
from typing import Callable, Final

from pycommons.io.path import Path
from pycommons.types import check_int_range

from moptipy.api.logging import FILE_SUFFIX
from moptipy.utils.logger import FileLogger, Logger

#: the default base directory for logging
_DEFAULT_DIR_NAME: Final[str] = "improvements"
#: the default base directory for logging
_DEFAULT_BASE_DIR: Final[str] = f"./{_DEFAULT_DIR_NAME}"
#: the default base name for log files
_DEFAULT_BASE_NAME: Final[str] = "improvement"


class ImprovementLogger:
    """A improvement logger."""

    def log_improvement(self, call: Callable[[Logger], None]) -> None:
        """
        Provide a logger to the callable to log an improvement.

        This method is invoked whenever an improvement was discovered during
        the optimization process. The method needs to create and provide a
        :class:`~moptipy.utils.logger.Logger` to the callable it receives as
        input.

        :param call: A :class:`Callable` to invoke with a logger to write the
            improvement information to.
        """
        raise NotImplementedError


class ImprovementLoggerFactory:
    """A factory for file improvement loggers."""

    def create(self, reference_path: str | None,
               reference_name: str | None) -> ImprovementLogger:
        """
        Create a new file improvement logger.

        :param reference_path: a path to a file used as reference for creating
            the logging base directory and file, or `None` if none is
            available
        :param reference_name: a name that can be used as reference for file
            names, or `None` if none is available
        :returns: the improvement logger
        """
        raise NotImplementedError


class FileImprovementLogger(ImprovementLogger):
    """
    A file-based improvement logger.

    This logger creates a new text file for each improvement.
    The improved solution is stored in the text file.
    The logger allows you to limit the number of retained files.
    If more improvements than the provided limit are created, then it will
    delete the oldest log file.
    """

    def __init__(self, log_dir: str | None = None,
                 log_base_name: str | None = None,
                 max_files: int | None = None) -> None:
        """
        Create the file improvement logger.

        :param log_dir: the directory to store the improvement logs in
        :param log_base_name: the base name of the improvement logs
        :param max_files: a limit for the number of log files to be retained,
            or `None` for unlimited
        """
        #: the logging directory
        self.__log_dir: Final[Path] = Path(
            _DEFAULT_BASE_DIR if log_dir is None else log_dir)
        self.__log_dir.ensure_dir_exists()

        log_base_name = _DEFAULT_BASE_NAME if log_base_name is None else (
            str.removesuffix(str.strip(log_base_name), "_"))
        if str.__len__(log_base_name) <= 0:
            raise ValueError("Log base name cannot be empty or just "
                             "consist of whitespace.")
        #: the log file base name
        self.__log_base_name: Final[str] = log_base_name
        #: the maximum number of files to keep alive
        self.__max_files: Final[int | None] = None if max_files is None \
            else check_int_range(max_files, "max_files", 1, 1_000_000_000_000)
        #: the file history
        self.__file_history: Final[list[Path] | None] = \
            None if max_files is None else []

        #: the file index
        self.__index: int = 0

    def log_improvement(self, call: Callable[[Logger], None]) -> None:
        """
        Log an improvement.

        :param call: the callable to do the logging
        """
        file: Path | None = None
        while True:
            self.__index += 1
            file = self.__log_dir.resolve_inside(
                f"{self.__log_base_name}_{self.__index}{FILE_SUFFIX}")
            if not file.ensure_file_exists():
                break

        call(FileLogger(file))
        if (self.__file_history is not None) and (
                self.__max_files is not None):
            self.__file_history.append(file)
            if list.__len__(self.__file_history) > self.__max_files:
                remove(self.__file_history.pop(0))


class FileImprovementLoggerFactory(ImprovementLoggerFactory):
    """A file-based improvement logger factory."""

    def __init__(self, base_dir: str | None = None,
                 log_base_name: str | None = None,
                 max_files: int | None = None) -> None:
        """
        Create the file improvement logger factory.

        :param base_dir: the base directory to store the improvement
            logs in, `None` for default
        :param log_base_name: the default base name of the improvement logs,
            `None` for default
        :param max_files: a limit for the number of log files to be retained,
            or `None` for unlimited
        """
        #: the logging directory
        self.__base_dir: Final[Path] = Path(
            _DEFAULT_BASE_DIR if base_dir is None else base_dir)

        log_base_name = _DEFAULT_BASE_NAME if log_base_name is None \
            else str.strip(log_base_name)
        if str.__len__(log_base_name) <= 0:
            raise ValueError("Log base name cannot be empty or just "
                             "consist of whitespace.")
        #: the log file base name
        self.__log_base_name: Final[str] = log_base_name
        #: the maximum number of files to keep alive
        self.__max_files: Final[int | None] = None if max_files is None \
            else check_int_range(max_files, "max_files", 1, 1_000_000_000_000)

    def create(self, reference_path: str | None,
               reference_name: str | None) -> ImprovementLogger:
        """
        Create a new file improvement logger.

        :param reference_path: a path to a file used as reference for creating
            the logging base directory and file, or `None` if none is
            available
        :param reference_name: a name that can be used as reference for file
            names, or `None` if none is available
        :returns: the improvement logger
        """
        log_dir: Path | None = None
        log_name: str | None = reference_name

        if reference_path is not None:
            pt: Path = Path(reference_path)
            log_name = (pt.basename().removesuffix(FILE_SUFFIX)
                        .removesuffix("_") + f"_{_DEFAULT_BASE_NAME}")
            log_dir = Path(str.removesuffix(
                pt, FILE_SUFFIX).removesuffix("_") + f"_{_DEFAULT_DIR_NAME}")

        if log_dir is None:
            log_dir = self.__base_dir
            log_dir.ensure_dir_exists()
            if log_name is not None:
                log_dir = log_dir.resolve_inside(str.removesuffix(
                    log_name, "_"))
        if log_name is None:
            log_name = self.__log_base_name

        return FileImprovementLogger(log_dir, log_name, self.__max_files)
