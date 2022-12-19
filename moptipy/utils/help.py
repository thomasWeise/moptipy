"""Print a help screen."""

import argparse
import os.path
import sys
from typing import Final

from moptipy.utils.path import _canonicalize_path
from moptipy.utils.types import type_error
from moptipy.version import __version__

#: The default argument parser for moptipy executables.
DEFAULT_ARGUMENTS: Final[argparse.ArgumentParser] = argparse.ArgumentParser(
    epilog="\n\n\u00a9 2022 Thomas Weise,\n"
           "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007,\n"
           "https://thomasweise.github.io/moptipy",
    add_help=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
DEFAULT_ARGUMENTS.add_argument(
    "--version", action="version", version=__version__)


#: The python interpreter in long form.
__INTERPRETER_LONG: Final[str] = _canonicalize_path(sys.executable)


def __get_python_interpreter_short() -> str:
    """
    Get the python interpreter.

    :returns: the fully-qualified path
    """
    bn = os.path.basename(__INTERPRETER_LONG)
    if bn.startswith("python3."):
        bn2 = bn[:7]
        interp2 = os.path.join(os.path.dirname(__INTERPRETER_LONG), bn2)
        if os.path.exists(interp2) and os.path.isfile(interp2):
            if _canonicalize_path(interp2) == __INTERPRETER_LONG:
                return bn2
    return bn


#: The python interpreter in short form.
__INTERPRETER_SHORT: Final[str] = __get_python_interpreter_short()
del __get_python_interpreter_short

#: the length of the base path
__BASE_PATH: Final[str] = _canonicalize_path(os.path.dirname(
    os.path.dirname(os.path.dirname(_canonicalize_path(__file__))))) + os.sep


def get_prog(file: str) -> str:
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

    # get the module
    module: str = _canonicalize_path(file)
    end: int = len(module)
    start: int = 0
    if module.endswith(".py"):
        end -= 3
    if module.startswith(__BASE_PATH):
        start += len(__BASE_PATH)
    module = module[start:end].replace(os.sep, ".")

    return f"{__INTERPRETER_SHORT} -m {module}"
