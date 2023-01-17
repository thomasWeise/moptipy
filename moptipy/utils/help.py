"""Print a help screen."""

import argparse
import os.path
import sys
from typing import Final

from moptipy.utils.path import _canonicalize_path
from moptipy.utils.types import type_error
from moptipy.version import __version__

#: The default argument parser for moptipy executables.
__DEFAULT_ARGUMENTS: Final[argparse.ArgumentParser] = argparse.ArgumentParser(
    epilog="Copyright\u00a0\u00a9\u00a02022\u00a0Thomas\u00a0WEISE, "
           "GNU\u00a0GENERAL\u00a0PUBLIC\u00a0LICENSE\u00a0Version\u00a03,"
           "\u00a029\u00a0June\u00a02007, "
           "https://thomasweise.github.io/moptipy, "
           "tweise@hfuu.edu.cn,\u00a0tweise@ustc.edu.cn",
    add_help=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
__DEFAULT_ARGUMENTS.add_argument(
    "--version", action="version", version=__version__)


def __get_python_interpreter_short() -> str:
    """
    Get the python interpreter.

    :returns: the fully-qualified path
    """
    inter: Final[str] = _canonicalize_path(sys.executable)
    bn = os.path.basename(inter)
    if bn.startswith("python3."):
        bn2 = bn[:7]
        interp2 = os.path.join(os.path.dirname(inter), bn2)
        if os.path.exists(interp2) and os.path.isfile(interp2) \
                and (_canonicalize_path(interp2) == inter):
            return bn2
    return bn


#: The python interpreter in short form.
__INTERPRETER_SHORT: Final[str] = __get_python_interpreter_short()
del __get_python_interpreter_short

#: the base path of the moptipy package
__BASE_PATH: Final[str] = _canonicalize_path(os.path.dirname(
    os.path.dirname(os.path.dirname(_canonicalize_path(__file__))))) + os.sep


def __get_prog(file: str) -> str:
    """
    Get the program as to be displayed by the help screen.

    The result of this function applied to the `__file__` special
    variable should be put into the `prog` argument of the constructor
    of :class:`argparse.ArgumentParser`.

    :param file: the calling python script
    :return: the program string
    """
    if not isinstance(file, str):
        raise type_error(file, "file", str)

    # get the module minus the base path and extension
    module: str = _canonicalize_path(file)
    end: int = len(module)
    start: int = 0
    if module.endswith(".py"):
        end -= 3
    if module.startswith(__BASE_PATH):
        start += len(__BASE_PATH)
    module = module[start:end].replace(os.sep, ".")

    return f"{__INTERPRETER_SHORT} -m {module}"


def argparser(file: str, description: str,
              epilog: str) -> argparse.ArgumentParser:
    """
    Create an argument parser with default settings.

    :param file: the `__file__` special variable of the calling script
    :param description: the description string
    :param epilog: the epilogue string
    :returns: the argument parser

    >>> ap = argparser(__file__, "This is a test program.", "This is a test.")
    >>> isinstance(ap, argparse.ArgumentParser)
    True
    >>> "Copyright" in ap.epilog
    True
    """
    if not isinstance(file, str):
        raise type_error(file, "file", str)
    if len(file) <= 3:
        raise ValueError(f"invalid file={file!r}.")
    if not isinstance(description, str):
        raise type_error(description, "description", str)
    if len(description) <= 12:
        raise ValueError(f"invalid description={description!r}.")
    if not isinstance(epilog, str):
        raise type_error(epilog, "epilog", str)
    if len(epilog) <= 10:
        raise ValueError(f"invalid epilog={epilog!r}.")
    return argparse.ArgumentParser(
        parents=[__DEFAULT_ARGUMENTS], prog=__get_prog(file),
        description=description.strip(),
        epilog=f"{epilog.strip()} {__DEFAULT_ARGUMENTS.epilog}",
        formatter_class=__DEFAULT_ARGUMENTS.formatter_class)
